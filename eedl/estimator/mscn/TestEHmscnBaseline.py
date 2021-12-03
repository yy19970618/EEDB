import os
import pickle
import time
import logging
from typing import Dict, Any, Tuple

import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader, dataset

from eedl.estimator.eedl.Dataflow import getQueryData, getTable, query_2_sql
from eedl.estimator.eedl.database import DBInterface
from eedl.estimator.eedl.memory import Memory
from eedl.estimator.mscn.model import SetConv
from eedl.estimator.estimator import Estimator, OPS
from eedl.estimator.utils import report_model, qerror, evaluate, run_test
from eedl.dataset.dataset import load_table
from eedl.workload.workload import load_queryset, load_labels, query_2_triple, Label
from eedl.constants import DEVICE, MODEL_ROOT, NUM_THREADS, DATA_ROOT, OUTPUT_ROOT
import myArgs

L = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.DEBUG)

L.addHandler(logging.StreamHandler())


class MyNumbers:
    def __iter__(self):
        self.a = 0
        return self

    def __next__(self):
        x = self.a
        self.a += 1
        return x


def Args(**params):
    y = myArgs.myArgs(**params)
    return y


def idx_to_onehot(idx, num_elements):
    onehot = np.zeros(num_elements, dtype=np.float32)
    onehot[idx] = 1.
    return onehot


def get_set_encoding(source_set, onehot=True):
    num_elements = len(source_set)
    source_list = list(source_set)
    # Sort list to avoid non-deterministic behavior
    source_list.sort()
    # Build map from s to i
    thing2idx = {s: i for i, s in enumerate(source_list)}
    # Build array (essentially a map from idx to s)
    idx2thing = [s for i, s in enumerate(source_list)]
    if onehot:
        thing2vec = {s: idx_to_onehot(i, num_elements) for i, s in enumerate(source_list)}
        return thing2vec, idx2thing
    return thing2idx, idx2thing


def normalize_labels(labels, min_val=None, max_val=None):
    # +1 to deal with 0 scenario
    labels = np.array([np.log(float(l.cardinality + 1)) for l in labels])
    if min_val is None:
        min_val = labels.min()
        L.info(f"min log(label): {min_val}")
    if max_val is None:
        max_val = labels.max()
        L.info(f"max log(label): {max_val}")
    labels_norm = (labels - min_val) / (max_val - min_val)
    # Threshold labels
    labels_norm = np.minimum(labels_norm, 1)
    labels_norm = np.maximum(labels_norm, 0)
    return labels_norm, min_val, max_val


def unnormalize_labels(labels_norm, min_val, max_val):
    labels_norm = np.array(labels_norm, dtype=np.float32)
    labels = (labels_norm * (max_val - min_val)) + min_val
    # -1 to deal with 0 scenario, need to restrict to >= 0
    return np.array(np.round(np.exp(labels) - 1), dtype=np.int64)


def get_sample_bitmap(sample, query):
    # do not need to convert [] to >= and <= here
    columns, operators, values = query_2_triple(query, with_none=False, split_range=False)
    bitmap = np.ones(len(sample), dtype=bool)
    for c, o, v in zip(columns, operators, values):
        bitmap &= OPS[o](sample[c], v)
    return [bitmap.astype(int)]


def encode_data(table, query, column2vec, op2vec):
    columns, operators, values = query_2_triple(query, with_none=False, split_range=True)
    predicates_enc = []
    for c, o, v in zip(columns, operators, values):
        norm_val = table.columns[c].normalize(v)
        pred_vec = []
        pred_vec.append(column2vec[c])
        pred_vec.append(op2vec[o])
        pred_vec.append(norm_val)
        pred_vec = np.hstack(pred_vec)
        predicates_enc.append(pred_vec)
    # for no predicate scenario
    if len(predicates_enc) == 0:
        predicates_enc.append(np.zeros((len(column2vec) + len(op2vec) + 1)))
    return predicates_enc


def encode_datas(table, queries, column2vec, op2vec):
    return [encode_data(table, q, column2vec, op2vec) for q in queries]


def load_dicts(table):
    # Get column name dict
    # we assume any column can have predicates
    column2vec, _ = get_set_encoding(table.data.columns)

    # Get operator name dict
    # NOTICE: [] should be converted to two operators: >= and <= later for mscn
    operators = set(['=', '>=', '<='])
    op2vec, _ = get_set_encoding(operators)

    # Get min max value for each column
    return column2vec, op2vec


def unnormalize_torch(vals, min_val, max_val):
    vals = (vals * (max_val - min_val)) + min_val
    return torch.exp(vals) - 1  # -1 since we +1 when normalize


def qerror_loss(preds, targets, min_val, max_val):
    errors = []
    preds = unnormalize_torch(preds, min_val, max_val)
    targets = unnormalize_torch(targets, min_val, max_val)

    for i in range(len(targets)):
        e = qerror(preds[i], targets[i])
        errors.append(e if torch.is_tensor(e) else torch.tensor([e], requires_grad=True, device=torch.device(DEVICE)))
    return torch.mean(torch.cat(errors))


def predict(model, data_loader, cuda):
    preds = []
    t_total = 0.

    model.eval()
    for batch_idx, data_batch in enumerate(data_loader):

        samples, predicates, targets, sample_masks, predicate_masks = data_batch

        if cuda:
            samples, predicates, targets = samples.cuda(), predicates.cuda(), targets.cuda()
            sample_masks, predicate_masks = sample_masks.cuda(), predicate_masks.cuda()
        samples, predicates, targets = Variable(samples), Variable(predicates), Variable(
            targets)
        sample_masks, predicate_masks = Variable(sample_masks), Variable(predicate_masks)

        t = time.time()
        outputs = model(samples, predicates, sample_masks, predicate_masks)
        t_total += time.time() - t

        for i in range(outputs.shape[0]):
            preds.append(outputs[i].cpu().item())

    return preds, t_total


def make_sample_tensor_mask(sample):
    # no need to pad since only for single table
    sample_tensor = np.vstack(sample)
    sample_mask = np.ones_like(sample_tensor).mean(1, keepdims=True)
    return sample_tensor, sample_mask


def make_predicate_tensor_mask(predicate, max_pred):
    predicate_tensor = np.vstack(predicate)
    num_pad = max_pred - predicate_tensor.shape[0]
    predicate_mask = np.ones_like(predicate_tensor).mean(1, keepdims=True)
    predicate_tensor = np.pad(predicate_tensor, ((0, num_pad), (0, 0)), 'constant')
    predicate_mask = np.pad(predicate_mask, ((0, num_pad), (0, 0)), 'constant')
    return predicate_tensor, predicate_mask


def make_dataset(samples, predicates, labels, max_pred):
    """Add zero-padding and wrap as tensor dataset."""
    sample_masks = []
    sample_tensors = []
    for sample in samples:
        sample_tensor, sample_mask = make_sample_tensor_mask(sample)
        sample_tensors.append(np.expand_dims(sample_tensor, 0))
        sample_masks.append(np.expand_dims(sample_mask, 0))
    sample_tensors = np.vstack(sample_tensors)
    sample_tensors = torch.FloatTensor(sample_tensors)
    sample_masks = np.vstack(sample_masks)
    sample_masks = torch.FloatTensor(sample_masks)
    L.debug(f'Sample tensor shape: {sample_tensors.shape}, mask shape: {sample_masks.shape}')

    predicate_masks = []
    predicate_tensors = []
    for predicate in predicates:
        predicate_tensor, predicate_mask = make_predicate_tensor_mask(predicate, max_pred)
        predicate_tensors.append(np.expand_dims(predicate_tensor, 0))
        predicate_masks.append(np.expand_dims(predicate_mask, 0))
    predicate_tensors = np.vstack(predicate_tensors)
    predicate_tensors = torch.FloatTensor(predicate_tensors)
    predicate_masks = np.vstack(predicate_masks)
    predicate_masks = torch.FloatTensor(predicate_masks)
    L.debug(f'Predicate tensor shape: {predicate_tensors.shape}, mask shape: {predicate_masks.shape}')

    target_tensor = torch.FloatTensor(labels)

    return dataset.TensorDataset(sample_tensors, predicate_tensors, target_tensor,
                                 sample_masks, predicate_masks)


class MSCN(Estimator):
    def __init__(self, model, model_name, samples, table, column2vec, op2vec, label_range):
        super(MSCN, self).__init__(table=table, model=model_name)
        self.model = model
        self.samples = samples
        self.column2vec = column2vec
        self.op2vec = op2vec
        self.minval = label_range[0]
        self.maxval = label_range[1]
        self.device = torch.device(DEVICE)
        self.model.to(self.device)
        self.model.eval()

    def query(self, query):
        sample_enc = get_sample_bitmap(self.samples, query)
        predicate_enc = encode_data(self.table, query, self.column2vec, self.op2vec)

        sample_tensor, sample_mask = make_sample_tensor_mask(sample_enc)
        predicate_tensor, predicate_mask = make_predicate_tensor_mask(predicate_enc, len(predicate_enc))

        sample_tensor = torch.FloatTensor(np.expand_dims(sample_tensor, axis=0)).to(self.device)
        sample_mask = torch.FloatTensor(np.expand_dims(sample_mask, axis=0)).to(self.device)
        predicate_tensor = torch.FloatTensor(np.expand_dims(predicate_tensor, axis=0)).to(self.device)
        predicate_mask = torch.FloatTensor(np.expand_dims(predicate_mask, axis=0)).to(self.device)

        start_stmp = time.time()
        with torch.no_grad():
            pred = self.model(sample_tensor, predicate_tensor, sample_mask, predicate_mask)
        dur_ms = (time.time() - start_stmp) * 1e3

        return unnormalize_labels(pred.cpu(), self.minval, self.maxval).item(), dur_ms


def TEST_MSCN_MODEL(seed, workload, params, table, queryset, labels):
    # uniform thread number
    torch.set_num_threads(NUM_THREADS)
    assert NUM_THREADS == torch.get_num_threads(), torch.get_num_threads()
    L.info(f"torch threads: {torch.get_num_threads()}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    # convert parameter dict of mscn
    L.info(f"params: {params}")
    args = Args(**params)
    # create model
    column2vec, op2vec = load_dicts(table)
    predicate_feats = len(column2vec) + len(op2vec) + 1
    model = SetConv(args.num_samples, predicate_feats, args.hid_units)

    model_path = MODEL_ROOT + "/" + table.dataset
    load_model_file = model_path + "/" + f"{table.version}_{workload}-{model.name()}_ep{args.epochs}_bs{args.bs}_{args.train_num // 1000}k-{seed}-base.pt"

    # 判断是否需要继续训练
    print(
        "Loading exist model: " + f"{table.version}_{workload}-{model.name()}_ep{args.epochs}_bs{args.bs}_{args.train_num // 1000}k-{seed}-base.pt")

    try:
        state = torch.load(load_model_file, map_location=DEVICE)
    except:
        print("Model file not exists!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return None, None
    model.load_state_dict(state['model_state_dict'])

    # materialize sample
    sample = table.data.sample(n=args.num_samples, random_state=seed)

    # Get feature encoding and proper normalization
    samples_valid = [get_sample_bitmap(sample, q) for q in queryset['valid']]
    predicates_train = [encode_data(table, q, column2vec, op2vec) for q in queryset['train']]
    predicates_valid = [encode_data(table, q, column2vec, op2vec) for q in queryset['valid']]
    label_norm, min_val, max_val = normalize_labels(labels['train'] + labels['valid'])
    labels_valid = label_norm[len(queryset['train']):]

    # Train model
    # NOTICE: do not record min max value for each column, make sure to load the same table when test
    state = {
        'seed': seed,
        'args': args,
        'device': DEVICE,
        'threads': torch.get_num_threads(),
        'dataset': table.dataset,
        'version': table.version,
        'workload': workload,
        'label_range': (min_val, max_val),
        'samples': sample
    }

    cuda = False if DEVICE == 'cpu' else True

    print("cuda?" + str(cuda))
    if cuda:
        model.cuda()

    max_pred = max(max([len(p) for p in predicates_train]), max([len(p) for p in predicates_valid]))
    valid_dataset = make_dataset(samples_valid, predicates_valid, labels=labels_valid, max_pred=max_pred)
    valid_data_loader = DataLoader(valid_dataset, batch_size=args.bs)

    model.train()
    preds_valid, _ = predict(model, valid_data_loader, cuda)
    # Unnormalize
    preds_valid_unnorm = unnormalize_labels(preds_valid, min_val, max_val)
    labels_valid_unnorm = unnormalize_labels(labels_valid, min_val, max_val)

    L.info("Q-Error on validation set:")
    _, metrics = evaluate(preds_valid_unnorm, labels_valid_unnorm)

    return metrics, state


def Test_all(dataset_name, num_samples, hid_units):
    global query, index
    '''
     这里需要修改！！！！！！！
     '''
    dataset_name = dataset_name
    metric_history = []
    '''
    ====================================================================================================================
    '''
    query_total = getQueryData(DATA_ROOT, dataset_name)
    with open(DATA_ROOT + "/" + dataset_name + "/workload/total_label.pkl", "rb") as f:
        label_total = pickle.load(f)
    table = getTable(dataset_name, "original")
    db = DBInterface(dataset_name + "_original")
    # 用来标记各个集合的大小
    train_size = 96000
    valid_size = 2000
    query_ptr = 0  # 用来标记全局的下一个查询应该是第几号，比如已经使用了全部数据集的10000条，那么其值就是10000

    data_pool = Memory()

    global_iter = iter(MyNumbers())
    for i in range(16000):
        query_ptr = next(global_iter) + 1
    model_arg = {"bs": 512, "train_num": train_size, "num_samples": num_samples, "hid_units": hid_units, "epochs": 60}
    '''
            ====================数据经验池存取合一操作==========================================================================
            '''
    for _ in range(train_size):
        sql_index = next(global_iter)
        query_ptr = sql_index + 1
        data_pool.save_store(query_total[sql_index], label_total[sql_index].cardinality)
        # 记录预处理时间

    # 获取用于训练的数据
    sample_data = data_pool.sample(train_size)
    sample_query = [query[0] for query in sample_data]
    sample_label = [Label(cardinality=int(query[1]), selectivity=int(query[1]) / db.table_row_num) for query in
                    sample_data]

    queries = {'train': sample_query}
    labels = {'train': sample_label}

    # 将"valid"的集合进行替换
    # 这里用于获取valid的数据，也就是说，这里的query和label是即将到来的查询的query和label
    sim_query = []
    sim_label = []
    for index in range(valid_size):
        sql_index = index + query_ptr
        sim_query.append(query_total[sql_index])
        sim_label.append(label_total[sql_index])

    # 将原来的验证集进行替换，这样使得模型更加合理
    queries["valid"] = sim_query
    labels["valid"] = sim_label
    '''
    ========================模型配置操作==============================================================================
    '''

    best_metric, state = TEST_MSCN_MODEL(100,
                                         dataset_name,
                                         model_arg,
                                         table,
                                         queries,
                                         labels)
    metric_history.append(best_metric)
    information = {"metric_history": metric_history, "model_arg": model_arg}
    with open(OUTPUT_ROOT + f"/information/{dataset_name}-test-valid2000-base.pkl", "wb") as f:
        pickle.dump(information, f)


if __name__ == '__main__':

    config = {"census13": [600, 4], "dmv11": [10000, 256], "forest10": [5000, 8], "power7": [10000, 16]}
    for dataset_name in [ "forest10"]:
        Test_all(dataset_name, config[dataset_name][0], config[dataset_name][1])
