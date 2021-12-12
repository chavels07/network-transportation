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
COST_PER_DIS = 0.5
VALUE_OF_TIME = 0.1
CHARGING_RATE = 0.1  # TODO:修正
REPLACE_FEE = 0.01
SPEED = 50
BIG_M = 500


def prompt_display(func):
    def inner(*args, **kwargs):
        frame_str = '=' * 15 + func.__name__ + '=' * 15
        print(frame_str)
        result = func(*args, **kwargs)
        print('=' * len(frame_str) + '\r\n')
        return result

    return inner


def read_data(path):
    data = pd.read_csv(path, encoding='utf8')
    return data.to_numpy()


class NetWork:
    __slots__ = ('G', 'solve_mode', 'loc', 'route')

    def __init__(self, nodes, edges):
        self.G, self.loc = self._init_graph(nodes, edges)
        self.solve_mode = 'Gurobi'  # 默认gurobi求解
        self.route = None

    @staticmethod
    def _init_graph(nodes: pd.DataFrame, edges: pd.DataFrame):
        g = nx.DiGraph()  # Warning: Graph create undirected graph
        append_nodes = list(map(lambda x: (x[1], {'charger': x[4], 'price': x[5]}), nodes.itertuples()))
        g.add_nodes_from(append_nodes)
        g.add_weighted_edges_from(
            [tuple(edge[1:]) for edge in edges.itertuples()])  # attribute weight refers to distance
        loc = dict(map(lambda x: (x[1], (x[2], x[3])), nodes.itertuples()))
        return g, loc

    @prompt_display
    def solve_single_by_gurobi(self):
        """
        version: 1.0
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
        dis_coeff = tupledict(map(lambda edge: ((edge[0], edge[1]), edge[2]['weight'] * COST_PER_DIS), g.edges.data()))
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
            self.route = self._get_shortest_path(x_var_dict)
            for edge in g.edges:
                print(edge, end='    ')
                print(x_var_dict[edge])
            for node in g.nodes:
                print(node, end='    ')
                print(s_var_dict[node])

    @prompt_display
    def solve_alternative_by_gurobi(self):
        """
        version: 1.0
        :return:
        """
        g = self.G
        m = Model('EVRP alternative')

        # add variables
        x_var_dict = m.addVars(g.edges, vtype=GRB.BINARY, name='x')
        q_var_dict = m.addVars(g.nodes, vtype=GRB.CONTINUOUS, name='q', ub=100, lb=0)
        e_var_dict = m.addVars(g.nodes, vtype=GRB.CONTINUOUS, name='e', ub=100, lb=0)
        r_var_dict = m.addVars(g.nodes, vtype=GRB.BINARY, name='r')
        s_var_dict = m.addVars(g.nodes, vtype=GRB.CONTINUOUS, name='s', ub=100, lb=0)

        # add objective function
        node_with_charger = [node[0] for node in g.nodes.data() if node[1]['charger'] == 1]
        price_coeff = tupledict(
            map(lambda node: (node[0], node[1]['price'] + VALUE_OF_TIME / CHARGING_RATE), g.nodes.data()))
        dis_coeff = tupledict(map(lambda edge: ((edge[0], edge[1]), edge[2]['weight'] * COST_PER_DIS), g.edges.data()))
        obj = e_var_dict.prod(price_coeff, node_with_charger) + REPLACE_FEE * r_var_dict.sum('*') + x_var_dict.prod(
            dis_coeff, '*')
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
                    expr_comp.append(x_var_dict[i, j] * quicksum([s_var_dict[i], - ELEC_CONS_RATE * d, q_var_dict[i]]))
                m.addConstr(quicksum(expr_comp) == s_var_dict[node], name='soc in node %d' % node)

            else:
                # origin node power constraint
                m.addConstr(s_var_dict[node] == ELEC_CAPACITY, name='')

            # charging constraint
            if para['charger'] == 1:
                m.addConstr(q_var_dict[node] <= ELEC_CAPACITY - s_var_dict[node], name='')

                # replacement indicator
                m.addConstr(q_var_dict[node] >= ELEC_CAPACITY - s_var_dict[node] - BIG_M * (1 - r_var_dict[node]),
                            name='')

                # charging fee
                m.addConstr(e_var_dict[node] <= BIG_M * (1 - r_var_dict[node]), name='')
                m.addConstr(e_var_dict[node] >= q_var_dict[node] - BIG_M * r_var_dict[node], name='')
            else:
                m.addConstr(q_var_dict[node] == 0, name='')

        m.update()

        print('约束数量%d' % len(m.getConstrs()))
        print('变量数量%d' % len(m.getVars()))
        m.optimize()
        if m.status == GRB.OPTIMAL:
            self.route = self._get_shortest_path(x_var_dict)

    def _get_shortest_path(self, x_var: tupledict):
        shortest_path_list = []
        for edge in self.G.edges:
            if x_var[edge].X > 0.99:
                shortest_path_list.append(edge)
        return shortest_path_list

    def network_display(self, mode='2dim'):
        if mode == '2dim':
            color = {
                -1: '#0000C6',
                0: '#6C6C6C',
                1: '#000000',
                2: '#FF0000'
            }
            node_color = [color[para['charger']] for _, para in self.G.nodes.data()]
            edge_color = ['#FF0000' if edge in self.route else '#000000' for edge in self.G.edges]
            edge_width = [2.5 if edge in self.route else 1.0 for edge in self.G.edges]
            nx.draw(self.G, pos=self.loc, node_size=50, node_color=node_color,
                    edge_color=edge_color)  # edge_width=edge_width,
            # nx.draw_networkx_edges(self.G, edge_color=)
            plt.show()
        if self.route is None:
            pass


if __name__ == '__main__':
    raw_node = pd.read_csv(NODE_PATH, encoding='utf8')
    raw_path = pd.read_csv(EDGE_PATH, encoding='utf8')
    raw_node['point'] = raw_node['point'].astype('int')
    raw_node['x'] = raw_node['x'].astype('int')
    raw_node['y'] = raw_node['y'].astype('int')
    raw_node['charger'] = raw_node['charger'].astype('int')
    raw_path['point1'] = raw_path['point1'].astype('int')
    raw_path['point2'] = raw_path['point2'].astype('int')
    network = NetWork(raw_node, raw_path)
    network.solve_alternative_by_gurobi()
    network.network_display()
