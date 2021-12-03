import psycopg2
import numpy as np
import matplotlib.pyplot as plt
import math
class DatabaseOp:
	table = [
		'customer',
		'lineitem',
		'nation',
		'orders',
		'part',
		'partsupp',
		'region',
		'supplier'
		]
	select_col=[[[13, 14, 15, 17, 18, 19]],
				[[1, 7], [9, 14, 16], [33, 36]],
				[[9, 20, 21], [29, 33, 34]],
				[[1], [11, 14, 15], [25, 26, 27], [30, 33], [48], [52, 53], [58]],
				[[13, 14, 15, 19]],
				[[1, 4], [11, 15, 19], [26], [30], [55, 58]],
				[[1, 4], [9, 10, 11, 14, 15], [25, 26, 27], [29, 30, 33], [38, 42], [52, 53], [55, 58]],
				[[9, 10, 11, 13, 14, 15], [25, 26], [29, 33], [38, 39], [47, 48, 50], [58]],
				[[1, 2, 3, 4, 5, 6], [8, 9, 14, 15, 17], [26], [29, 30, 33]],
				[[25, 26], [47, 49, 50], [55, 58]],
				[[9, 19, 20, 21, 23], [28, 29]],
				[[1], [29, 30], [37]],
				[[10, 14, 15, 19], [38, 42]],
				[[38, 41, 42, 43], [47, 48, 51]],
				[[1, 2], [9, 13], [29, 30, 32, 33]],
				[[10, 13, 14, 15, 22, 23], [38, 41, 43, 44]],
				[[9, 11, 20, 21], [25, 26], [29, 31], [48], [56, 58]],
				[[1, 5, 6], [30]]]
	column=[ 'c_custkey',       #0
  'c_name',#1
  'c_address',#2
  'c_nationkey',#3
  'c_phone',#4
  'c_acctbal',#5
  'c_mktsegment' ,#6
  'c_comment' ,#7
   'l_orderkey',#8
  'l_partkey',#9
  'l_suppkey',#10
  'l_linenumber',#11
  'l_quantity',#12
  'l_extendedprice',#13
  'l_discount',#14
  'l_tax',#15
  'l_returnflag',#16
  'l_linestatus' ,#17
  'l_shipdate',#18
  'l_commitdate',#19
  'l_receiptdate',#20
  'l_shipinstruct',#21
  'l_shipmode',#22
  'l_comment',#23
  'n_nationkey',#24
  'n_name',#25
  'n_regionkey',#26
  'n_comment',#27
   'o_orderkey',#28
  'o_custkey',#29
  'o_orderstatus' ,#30
  'o_totalprice',#31
  'o_orderdate',#32
  'o_orderpriority',#33
  'o_clerk',#34
  'o_shippriority',#35
  'o_comment' ,#36
  'p_partkey',#37
  'p_name',#38
  'p_mfgr',#39
  'p_brand',#40
  'p_type',#41
  'p_size',#42
  'p_container',#43
  'p_retailprice',#44
  'p_comment',#45
    'ps_partkey',#46
  'ps_suppkey',#47
  'ps_availqty',#48
  'ps_supplycost',#49
  'ps_comment',#50
   'r_regionkey',#51
  'r_name',#52
  'r_comment',#53
   's_suppkey',#54
  's_name',#55
  's_address',#56
  's_nationkey',#57
  's_phone',#58
  's_acctbal' ,#59
  's_comment'#60
  ]


	query=[
        ( "select "
	 "l_returnflag, "
	 "l_linestatus, "
	 "sum(l_quantity) as sum_qty, "
	 "sum(l_extendedprice) as sum_base_price, "
	 "sum(l_extendedprice * (1 - l_discount)) as sum_disc_price, "
	 "sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) as sum_charge, "
	 "avg(l_quantity) as avg_qty, "
	 "avg(l_extendedprice) as avg_price, "
	 "avg(l_discount) as avg_disc, "
	 "count(*) as count_order "
 "from "
	 "lineitem "
 "where "
	 "l_shipdate <= date '1998-12-01' - interval '90' day "
 "group by "
	 "l_returnflag, "
	 "l_linestatus "
 "order by "
	 "l_returnflag, "
	 "l_linestatus; "
	 
 
	),#13,14,15,17,18,19

	( "select "
	 "l_orderkey, "
	 "sum(l_extendedprice * (1 - l_discount)) as revenue, "
	 "o_orderdate, "
	 "o_shippriority "
 "from "
	 "customer, "
	 "orders, "
	 "lineitem "
 "where "
	 "c_mktsegment = 'BUILDING' "
	 "and c_custkey = o_custkey "
	 "and l_orderkey = o_orderkey "
	 "and o_orderdate < date '1995-03-15' "
	 "and l_shipdate > date '1995-03-15' "
 "group by "
	 "l_orderkey, "
	 "o_orderdate, "
	 "o_shippriority "
 "order by "
	 "revenue desc, "
	 "o_orderdate; "
	 
 
		),#9,16,14,33,36,7,1
		( "select "
	 "o_orderpriority, "
	 "count(*) as order_count "
 "from "
	 "orders "
 "where "
	 "o_orderdate >= date '1993-07-01' "
	 "and o_orderdate < date '1993-07-01' + interval '3' month "
	 "and exists ( "
		 "select "
			 "* "
		 "from "
			 "lineitem "
		 "where "
			 "l_orderkey = o_orderkey "
			 "and l_commitdate < l_receiptdate "
	 ") "
 "group by "
	 "o_orderpriority "
 "order by "
	 "o_orderpriority; "
	 
 
			),#34,33,9,29,20,21
		(
			 "select "
	 "n_name, "
	 "sum(l_extendedprice * (1 - l_discount)) as revenue "
 "from "
 "	customer, "
 "	orders, "
 "	lineitem, "
 "	supplier, "
 "	nation, "
 "	region "
 "where "
 "	c_custkey = o_custkey "
 "	and l_orderkey = o_orderkey "
 "	and l_suppkey = s_suppkey "
 "	and c_nationkey = s_nationkey "
 "	and s_nationkey = n_nationkey "
 "	and n_regionkey = r_regionkey "
 "	and r_name = 'ASIA' "
 "	and o_orderdate >= date '1994-01-01' "
 "	and o_orderdate < date '1994-01-01' + interval '1' year "
 "group by "
 "	n_name "
 "order by "
 "	revenue desc; "
 
 
			),#26,14,15,1,30,11,48,58,25,27,52,53,33
			(
				 "select "
	 "sum(l_extendedprice * l_discount) as revenue "
 "from "
 "	lineitem "
 "where "
 "	l_shipdate >= date '1994-01-01' "
 "	and l_shipdate < date '1994-01-01' + interval '1' year "
	 "and l_discount between .06 - 0.01 and .06 + 0.01 "
 "	and l_quantity < 24; "
 
 
				),#14,15,19,13
				(
					 "select "
	 "supp_nation, "
	 "cust_nation, "
	 "l_year, "
	 "sum(volume) as revenue "
 "from "
	 "( "
		 "select "
			 "n1.n_name as supp_nation, "
			 "n2.n_name as cust_nation, "
			 "extract(year from l_shipdate) as l_year, "
			 "l_extendedprice * (1 - l_discount) as volume "
		 "from "
			 "supplier, "
			 "lineitem, "
			 "orders, "
			 "customer, "
			 "nation n1, "
			 "nation n2 "
		 "where "
			 "s_suppkey = l_suppkey "
			 "and o_orderkey = l_orderkey "
			 "and c_custkey = o_custkey "
			 "and s_nationkey = n1.n_nationkey "
			 "and c_nationkey = n2.n_nationkey "
			 "and ( "
			 "	(n1.n_name = 'FRANCE' and n2.n_name = 'GERMANY') "
			 "	or (n1.n_name = 'GERMANY' and n2.n_name = 'FRANCE') "
			 ") "
		 "	and l_shipdate between date '1995-01-01' and date '1996-12-31' "
	 ") as shipping "
 "group by "
 "	supp_nation, "
	 "cust_nation, "
 "	l_year "
 "order by "
 "	supp_nation, "
 "	cust_nation, "
 "	l_year; "
 
 
					),#26,19,15,55,11,1,30,58,30,4
					(
 "						select "
 "	o_year, "
 "	sum(case "
 "		when nation = 'BRAZIL' then volume "
 "		else 0 "
 "	end) / sum(volume) as mkt_share "
 "from "
 "	( "
 "		select "
 "			extract(year from o_orderdate) as o_year, "
 "			l_extendedprice * (1 - l_discount) as volume, "
 "			n2.n_name as nation "
 "		from "
 "			part, "
 "			supplier, "
 "			lineitem, "
 "			orders, "
 "			customer, "
 "			nation n1, "
 "			nation n2, "
 "			region "
 "		where "
 "			p_partkey = l_partkey "
 "			and s_suppkey = l_suppkey "
 "			and l_orderkey = o_orderkey "
 "			and o_custkey = c_custkey "
 "			and c_nationkey = n1.n_nationkey "
 "			and n1.n_regionkey = r_regionkey "
 "			and r_name = 'AMERICA' "
 "			and s_nationkey = n2.n_nationkey "
 "			and o_orderdate between date '1995-01-01' and date '1996-12-31' "
 "			and p_type = 'ECONOMY ANODIZED STEEL' "
 "	) as all_nations "
 "group by "
 "	o_year "
 "order by "
 "	o_year; "
 
 
						),#33,14,15,26,38,10,55,11,9,29,30,1,4,25,27,52,53,58,42
						(
						 "	select "
	 "nation, "
 "	o_year, "
 "	sum(amount) as sum_profit "
 "from "
 "	( "
	 "	select "
	 "		n_name as nation, "
	 "		extract(year from o_orderdate) as o_year, "
	 "		l_extendedprice * (1 - l_discount) - ps_supplycost * l_quantity as amount "
 "		from "
 "			part, "
 "			supplier, "
 "			lineitem, "
 "			partsupp, "
 "			orders, "
 "			nation "
 "		where "
 "			s_suppkey = l_suppkey "
 "			and ps_suppkey = l_suppkey "
 "			and ps_partkey = l_partkey "
 "			and p_partkey = l_partkey "
 "			and o_orderkey = l_orderkey "
 "			and s_nationkey = n_nationkey "
 "			and p_name like '%green%' "
 "	) as profit "
 "group by "
 "	nation, "
 "	o_year "
 "order by "
 "	nation, "
 "	o_year desc; "
 
 
							),#26,33,14,15,50,13,48,11,47,10,38,29,9,58,25,39
							(
								 "select "
	 "c_custkey, "
	 "c_name, "
	 "sum(l_extendedprice * (1 - l_discount)) as revenue, "
	 "c_acctbal, "
	 "n_name, "
	 "c_address, "
	 "c_phone, "
	 "c_comment "
 "from "
	 "customer, "
	 "orders, "
	 "lineitem, "
	 "nation "
 "where "
	 "c_custkey = o_custkey "
	 "and l_orderkey = o_orderkey "
	 "and o_orderdate >= date '1993-10-01' "
	 "and o_orderdate < date '1993-10-01' + interval '3' month "
	 "and l_returnflag = 'R' "
 "	and c_nationkey = n_nationkey "
 "group by "
 "	c_custkey, "
 "	c_name, "
 "	c_acctbal, "
 "	c_phone, "
 "	n_name, "
	 "c_address, "
 "	c_comment "
 "order by "
 "	revenue desc; "
 
								),#1,2,14,15,3,6,26,5,8,30,9,29,33,17,4
								(
									 "select "
	 "ps_partkey, "
	 "sum(ps_supplycost * ps_availqty) as value "
 "from "
	 "partsupp, "
	 "supplier, "
	 "nation "
 "where "
	 "ps_suppkey = s_suppkey "
	 "and s_nationkey = n_nationkey "
	 "and n_name = 'GERMANY' "
 "group by "
	 "ps_partkey having "
	 "	sum(ps_supplycost * ps_availqty) > ( "
		 "	select "
		 "		sum(ps_supplycost * ps_availqty) * 0.0001000000 "
		 "	from "
		 "		partsupp, "
		 "		supplier, "
		 "		nation "
	 "		where "
	 "			ps_suppkey = s_suppkey "
	 "			and s_nationkey = n_nationkey "
	 "			and n_name = 'GERMANY' "
 "		) "
 "order by "
	 "value desc; "
	 
 
									),#47,50,49,55,58,25,26
									(
								 "		select "
 "	l_shipmode, "
 "	sum(case "
	 "	when o_orderpriority = '1-URGENT' "
	 "		or o_orderpriority = '2-HIGH' "
	 "		then 1 "
	 "	else 0 "
 "	end) as high_line_count, "
 "	sum(case "
 "		when o_orderpriority <> '1-URGENT' "
 "			and o_orderpriority <> '2-HIGH' "
 "			then 1 "
 "		else 0 "
 "	end) as low_line_count "
 "from "
 "	orders, "
 "	lineitem "
 "where "
 "	o_orderkey = l_orderkey "
	 "and l_shipmode in ('MAIL', 'SHIP') "
	 "and l_commitdate < l_receiptdate "
	 "and l_shipdate < l_commitdate "
 "	and l_receiptdate >= date '1994-01-01' "
 "	and l_receiptdate < date '1994-01-01' + interval '1' year "
 "group by "
 "	l_shipmode "
 "order by "
 "	l_shipmode; "
 
 
										),#23,28,29,9,20,21,19
										(
											 "select "
	 "c_count, "
	 "count(*) as custdist "
 "from "
	 "( "
	 "	select "
	 "		c_custkey, "
	 "		count(o_orderkey) "
	 "	from "
	 "		customer left outer join orders on "
	 "			c_custkey = o_custkey "
	 "			and o_comment not like '%special%requests%' "
 "		group by "
 "			c_custkey "
 "	) as c_orders (c_custkey, c_count) "
 "group by "
 "	c_count "
 "order by "
	 "custdist desc, "
 "	c_count desc; "
 
 
											),#1,29,30,37
											(
												 "select "
	 "100.00 * sum(case "
		 "when p_type like 'PROMO%' "
			 "then l_extendedprice * (1 - l_discount) "
		 "else 0 "
	 "end) / sum(l_extendedprice * (1 - l_discount)) as promo_revenue "
 "from "
	 "lineitem, "
	 "part "
 "where "
 "	l_partkey = p_partkey "
 "	and l_shipdate >= date '1995-09-01' "
 "	and l_shipdate < date '1995-09-01' + interval '1' month; "
 
 
												),#42,14,15,10,38,19
													(
														 "select "
	 "p_brand, "
	 "p_type, "
	 "p_size, "
	 "count(distinct ps_suppkey) as supplier_cnt "
 "from "
	 "partsupp, "
 "	part "
 "where "
	 "p_partkey = ps_partkey "
	 "and p_brand <> 'Brand#45' "
	 "and p_type not like 'MEDIUM POLISHED%' "
	 "and p_size in (49, 14, 23, 45, 19, 3, 36, 9) "
	 "and ps_suppkey not in ( "
		 "select "
			 "s_suppkey "
		 "from "
			 "supplier "
		 "where "
			 "s_comment like '%Customer%Complaints%' "
	 ") "
 "group by "
	 "p_brand, "
	 "p_type, "
	 "p_size "
 "order by "
	 "supplier_cnt desc, "
	 "p_brand, "
	 "p_type, "
	 "p_size; "
 
 
														),#41,42,43,48,38,47,48,51
														(
 "														select "
 "	c_name, "
 "	c_custkey, "
 "	o_orderkey, "
 "	o_orderdate, "
 "	o_totalprice, "
 "	sum(l_quantity) "
 "from "
 "	customer, "
 "	orders, "
 "	lineitem "
 "where "
 "	o_orderkey in ( "
 "		select "
 "			l_orderkey "
 "		from "
 "			lineitem "
 "		group by "
 "			l_orderkey having "
 "				sum(l_quantity) > 300 "
 "	) "
 "	and c_custkey = o_custkey "
 "	and o_orderkey = l_orderkey "
 "group by "
 "	c_name, "
 "	c_custkey, "
 "	o_orderkey, "
 "	o_orderdate, "
 "	o_totalprice "
 "order by "
 "	o_totalprice desc, "
 "	o_orderdate; "
 

	),#2,1,29,33,32,13,9,30
	(
 "select "
 "	sum(l_extendedprice* (1 - l_discount)) as revenue "
 "from "
 "	lineitem, "
 "	part "
 "where "
 "	( "
 "		p_partkey = l_partkey "
 "		and p_brand = 'Brand#12' "
 "		and p_container in ('SM CASE', 'SM BOX', 'SM PACK', 'SM PKG') "
 "		and l_quantity >= 1 and l_quantity <= 1 + 10 "
 "		and p_size between 1 and 5 "
 "		and l_shipmode in ('AIR', 'AIR REG') "
 "		and l_shipinstruct = 'DELIVER IN PERSON' "
 "	) "
 "	or "
 "	( "
 "		p_partkey = l_partkey "
 "		and p_brand = 'Brand#23' "
 "		and p_container in ('MED BAG', 'MED BOX', 'MED PKG', 'MED PACK') "
 "		and l_quantity >= 10 and l_quantity <= 10 + 10 "
 "		and p_size between 1 and 10 "
 "		and l_shipmode in ('AIR', 'AIR REG') "
 "		and l_shipinstruct = 'DELIVER IN PERSON' "
 "	) "
 "	or "
 "	( "
 "		p_partkey = l_partkey "
 "		and p_brand = 'Brand#34' "
 "		and p_container in ('LG CASE', 'LG BOX', 'LG PACK', 'LG PKG') "
 "		and l_quantity >= 20 and l_quantity <= 20 + 10 "
 "		and p_size between 1 and 15 "
 "		and l_shipmode in ('AIR', 'AIR REG') "
 "		and l_shipinstruct = 'DELIVER IN PERSON' "
 "	); "
 
 


	),#14,15,38,10,41,44,13,43,22,23
															(
 "																select "
 "	s_name, "
 "	count(*) as numwait "
 "from "
 "	supplier, "
 "	lineitem l1, "
 "	orders, "
 "	nation "
 "where "
 "	s_suppkey = l1.l_suppkey "
 "	and o_orderkey = l1.l_orderkey "
 "	and o_orderstatus = 'F' "
 "	and l1.l_receiptdate > l1.l_commitdate "
 "	and exists ( "
 "		select "
 "			* "
 "		from "
 "			lineitem l2 "
 "		where "
 "			l2.l_orderkey = l1.l_orderkey "
 "			and l2.l_suppkey <> l1.l_suppkey "
 "	) "
 "	and not exists ( "
 "		select "
 "			* "
 "		from "
 "			lineitem l3 "
 "		where "
 "			l3.l_orderkey = l1.l_orderkey "
 "			and l3.l_suppkey <> l1.l_suppkey "
 "			and l3.l_receiptdate > l3.l_commitdate "
 "	) "
 "	and s_nationkey = n_nationkey "
 "	and n_name = 'SAUDI ARABIA' "
 "group by "
 "	s_name "
 "order by "
 "	numwait desc, "
 "	s_name; "
 
),#56,48,11,29,9,31,21,20,58,25,26
	(
 "																	select "
 "	cntrycode, "
 "	count(*) as numcust, "
 "	sum(c_acctbal) as totacctbal "
 "from "
 "	( "
 "		select "
 "			substring(c_phone from 1 for 2) as cntrycode, "
 "			c_acctbal "
 "		from "
 "			customer "
 "		where "
 "			substring(c_phone from 1 for 2) in "
 "				('13', '31', '23', '29', '30', '18', '17') "
 "			and c_acctbal > ( "
 "				select "
 "					avg(c_acctbal) "
 "				from "
 "					customer "
 "				where "
 "					c_acctbal > 0.00 "
 "					and substring(c_phone from 1 for 2) in "
 "						('13', '31', '23', '29', '30', '18', '17') "
 "			) "
 "			and not exists ( "
 "				select "
 "					* "
 "				from "
 "					orders "
 "				where "
 "					o_custkey = c_custkey "
 "			) "
 "	) as custsale "
 "group by "
 "	cntrycode "
 "order by "
 "	cntrycode; "
 
 
	)#6,5,30,1
    ]
	pir=[37,46,54,47,57,24,26,51,52,
	  6,0,29,8,28,32,18,
	  19,20,3,18,12,38,22,40,41,42,43,21
	  ]
	select_num=[]
	index_size=0
	index_id={}
	cost_id=0
	index_num=0
	index_num_ar=[]
	query_cost=[]
	def __init__(self,exp_store):
		for i in range(18):
			self.query_cost.append([])
			self.select_num.append(0)
		for i in range(100):
			self.select_num[exp_store[i]]+=1

		self.conn = psycopg2.connect(database="tpcd", user="postgres",
		password="postgres", host="127.0.0.1", port="5432")
		self.cursor = self.conn.cursor()
		self.base_cost=[]
		for i in range(18):
				s="explain "+self.query[i]
				self.cursor.execute(s)
				rows = self.cursor.fetchall()
				str1=rows[0]
				strs=str1[0].split("cost=")
				s1=strs[1].split("..")
				c1=s1[0]
				s2=s1[1].split(" rows")
				c2=s2[0]
				c =float(c2)/1000
				self.base_cost.append(c)
	def select(self,sql):
		self.cursor.execute(sql)
		rows = self.cursor.fetchall()
		return rows
	def findTable(self,action):
		tb=0
		if action<8:
			tb=0
		elif action<24:
			tb=1
		elif action<28:
			tb=2
		elif action<37:
			tb=3

		elif action<46:
			tb=4
		elif action<51:
			tb=5
		elif action<54:
			tb=6
		else:
			tb=7
		return tb
	def addIndex(self,action_state,action,new):
		old_st=''.join('%s' %id for id in action_state)
		if action>=61:
			return
		action_state[action]=1
		st=''.join('%s' %id for id in action_state)
		index_list=[]
		for i in range(61):
			if action_state[i]==1 and i in self.pir:
				index_list.append(i)
		for i in range(61):
			if action_state[i]==1 and i not in self.pir:
				index_list.append(i)
		s="SELECT * FROM hypopg_create_index('CREATE INDEX ON "+self.table[self.findTable(index_list[0])]+"("
		for i in range(len(index_list)-1):
			s=s+self.column[index_list[i]]+","
		s=s+self.column[index_list[len(index_list)-1]]+")'); "
		print(s)
		self.cursor.execute(s)
		rows = self.cursor.fetchall()
		self.index_id[st]=rows[0][0]
#		q_s="SELECT pg_size_pretty(hypopg_relation_size("+str(rows[0][0])+")) ;"
#		self.cursor.execute(q_s)
#		rows = self.cursor.fetchall()
#		sizes=rows[0][0].split(" ")
#		size=sizes[0]
#		type=sizes[1]
#		if type=="kB":
#			size=float(size)/1000
#		self.index_size+=float(size)
		if new == 0:
			if old_st not in self.index_id:
				print("错误%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
				return None
			d_id=self.index_id[old_st]
			del self.index_id[old_st]
			self.cursor.execute("select hypopg_drop_index("+str(d_id)+");")
		return rows
	def dropIndex(self,action_state):
		old_st=''.join('%s' %id for id in action_state)
		if old_st not in self.index_id:
			print("错误%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
			return None
		s="select hypopg_drop_index("+str(self.index_id[old_st])+");"
		self.cursor.execute(s)
		rows = self.cursor.fetchall()
#		q_s="SELECT pg_size_pretty(hypopg_relation_size("+str(self.index_id[old_st])+")) ;"
#		self.cursor.execute(q_s)
#		rows = self.cursor.fetchall()
#		sizes=rows[0][0].split(" ")
#		size=sizes[0]
#		type=sizes[1]
#		if type=="kB":
#			size=float(size)/1000
#		self.index_size-=float(size)
		del self.index_id[old_st]
		
		return rows
	def getCost(self,delete_num,num):
		s_cost=[]
		a=np.random.randint(0, 18)
#		if num<5000:
#			while a<0 or a>=18:
#				a=np.random.normal(loc=4, scale=4)
#		else:
#			while a<0 or a>=18:
#				a=np.random.normal(loc=13, scale=4)
#		a = int(a)

		self.select_num[a]+=1
		self.select_num[delete_num]-=1
		for i in range(18):
			s="explain "+self.query[i]
			self.cursor.execute(s)
			rows = self.cursor.fetchall()
			str1=rows[0]
			strs=str1[0].split("cost=")
			s1=strs[1].split("..")
			c1=s1[0]
			s2=s1[1].split(" rows")
			c2=s2[0]
			c =float(c2)/1000
			s_cost.append(c)
		sum=0
		for i in range(18):
			sum=sum+(self.base_cost[i]-s_cost[i])/self.base_cost[i]*self.select_num[i]
		if self.index_size==0:
			return sum,a,s_cost
		else:
			return sum,a,s_cost
	def find(self):
		self.cursor.execute("select * from hypopg_list_indexes;")
		rows = self.cursor.fetchall()
		return rows
