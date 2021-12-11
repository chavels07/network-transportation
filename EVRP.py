from gurobipy import *
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

NODE_PATH = 'node.csv'
EDGE_PATH = 'edge.csv'
ELEC_CAPACITY = 100  # 电池容量
ELEC_CONS_RATE = 0.2  # 每公里电量消耗率
ELEC_MIN = 5  # 电池最小剩余量
VALUE_OF_TIME = 0.5
SPEED = 10
BIG_M = 500


def prompt_display(func):
    def inner(*args, **kwargs):
        frame_str = '='*15 + func.__name__ + '='*15
        print(frame_str)
        result = func(*args, **kwargs)
        print('=' * len(frame_str) + '\r\n')
        return result

    return inner


def read_data(path):
    data = pd.read_csv(path, encoding='utf8')
    return data.to_numpy()


class NetWork:
    def __init__(self, nodes, edges):
        self.G = self._init_graph(nodes, edges)
        self.solve_mode = 'Gurobi'  # 默认gurobi求解

    @staticmethod
    def _init_graph(nodes: pd.DataFrame, edges: pd.DataFrame):
        g = nx.DiGraph()  # Warning: Graph create undirected graph
        append_nodes = list(map(lambda row: (row[1], {'charger': row[4], 'price': row[5]}), nodes.itertuples()))
        g.add_nodes_from(append_nodes)
        g.add_weighted_edges_from(
            [tuple(edge[1:]) for edge in edges.itertuples()])  # attribute weight refers to distance
        return g

    @prompt_display
    def solve_by_gurobi(self):
        """

        :return:
        """
        g = self.G
        m = Model('EVRP')

        # add variables
        x_var_dict = m.addVars(g.edges, vtype=GRB.BINARY, name='x')
        e_var_dict = m.addVars(g.nodes, vtype=GRB.CONTINUOUS, name='e', ub=100, lb=ELEC_MIN - 100)
        s_var_dict = m.addVars(g.nodes, vtype=GRB.CONTINUOUS, name='s', ub=100, lb=0)

        # add objective function
        node_with_charger = [node[0] for node in g.nodes.data() if node[1]['charger'] == 1]
        price_coeff = tupledict(map(lambda node: (node[0], node[1]['price']), g.nodes.data()))
        dis_coeff = tupledict(map(lambda edge: ((edge[0], edge[1]), edge[2]['weight'] * VALUE_OF_TIME), g.edges.data()))
        obj = e_var_dict.prod(price_coeff, node_with_charger) + x_var_dict.prod(dis_coeff, '*')
        m.setObjective(obj, GRB.MINIMIZE)

        # add constraints
        for node, para in g.nodes.data():
            # unique flow constraints
            if para['charger'] == -1:
                right_hand_side_value = 1
            elif para['charger'] == 2:
                right_hand_side_value = -1
            else:
                right_hand_side_value = 0
            flow_in_node_cstr = LinExpr()
            for i, j in g.out_edges(node):
                flow_in_node_cstr += x_var_dict[i, j]
            for i, j in g.in_edges(node):
                flow_in_node_cstr -= x_var_dict[i, j]
            m.addConstr(flow_in_node_cstr == right_hand_side_value, name='flow in node %d' % node)

            if para['charger'] >= 0:
                # lower bound of SOC
                m.addConstr(x_var_dict.sum('*', node) * ELEC_MIN <= s_var_dict[node],
                            name='minimum soc in node %d' % node)
                m.addConstr(x_var_dict.sum('*', node) * BIG_M >= s_var_dict[node], name='')

                # equation of SOC
                expr_comp = []
                for i, j, d in g.in_edges(node, data='weight'):
                    expr_comp.append(x_var_dict[i, j] * quicksum([s_var_dict[i], - ELEC_CONS_RATE * d, e_var_dict[i]]))
                m.addConstr(quicksum(expr_comp) == s_var_dict[node], name='soc in node %d' % node)
            else:
                # origin node power constraint
                m.addConstr(s_var_dict[node] == 100, name='')

            # charging constraint
            if para['charger'] == 1:
                m.addConstr(e_var_dict[node] <= ELEC_CAPACITY - s_var_dict[node], name='')
                m.addConstr(e_var_dict[node] >= -(s_var_dict[node] - ELEC_MIN), name='')
            else:
                m.addConstr(e_var_dict[node] == 0, name='')

        m.update()

        print('约束数量%d' % len(m.getConstrs()))
        print('变量数量%d' % len(m.getVars()))
        m.optimize()
        if m.status == GRB.OPTIMAL:
            for edge in g.edges:
                print(edge, end='    ')
                print(x_var_dict[edge])
            for node in g.nodes:
                print(node, end='    ')
                print(s_var_dict[node])

    def network_display(self):
        nx.draw_networkx(self.G)
        plt.show()


if __name__ == '__main__':
    raw_node = pd.read_csv(NODE_PATH, encoding='utf8')
    raw_path = pd.read_csv(EDGE_PATH, encoding='utf8')
    raw_node['point'] = raw_node['point'].astype('int')
    raw_node['charger'] = raw_node['charger'].astype('int')
    raw_path['point1'] = raw_path['point1'].astype('int')
    raw_path['point2'] = raw_path['point2'].astype('int')
    network = NetWork(raw_node, raw_path)
    network.solve_by_gurobi()

