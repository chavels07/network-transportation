from typing import List
from gurobipy import *
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

NODE_PATH = 'node.csv'
EDGE_PATH = 'edge.csv'
ELEC_CAPACITY = 100  # 电池容量
ELEC_CONS_RATE = 0.2  # 每公里电量消耗率
ELEC_MIN = 5  # 电池最小剩余量
COST_PER_DIS = 0.5
VALUE_OF_TIME = 0.5
CHARGING_RATE = 0.1  # TODO:修正
REPLACE_FEE = 0.01
SPEED = 50
BIG_M = 200


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
    __slots__ = ('G', 'mod_G', 'loc', 'mod_loc', 'fold_path', 'route')

    def __init__(self, nodes, edges):
        self.G, self.loc, self.mod_G, self.mod_loc, self.fold_path = self._init_graph(nodes, edges)
        self.route = None

    @staticmethod
    def _init_graph(nodes: pd.DataFrame, edges: pd.DataFrame):
        g = nx.DiGraph()  # Warning: Graph create undirected graph
        mod_g = nx.DiGraph()
        append_nodes = list(map(lambda x: (x[1], {'charger': x[4], 'price': x[5]}), nodes.itertuples()))
        g.add_nodes_from(append_nodes)
        g.add_weighted_edges_from(
            [tuple(edge[1:]) for edge in edges.itertuples()])  # attribute weight refers to distance
        loc = dict(map(lambda x: (x[1], (x[2], x[3])), nodes.itertuples()))

        # construct modified graph
        key_nodes = nodes.loc[nodes['charger'] != 0].reset_index(drop=True)
        append_key_nodes = list(map(lambda x: (x[1], {'charger': x[4], 'price': x[5]}), key_nodes.itertuples()))
        mod_g.add_nodes_from(append_key_nodes)
        key_edges = []
        fold_path = {}
        for i in range(len(mod_g.nodes)):
            for j in range(i + 1, len(mod_g.nodes)):
                source_node = key_nodes['point'][i]
                target_node = key_nodes['point'][j]
                if nx.has_path(g, source=source_node, target=target_node):
                    # 小于电车单次最远行驶距离的可充电节点才可连通
                    modified_dis = nx.shortest_path_length(g, source=source_node, target=target_node, weight='weight')
                    if modified_dis <= (ELEC_CAPACITY - ELEC_MIN) / ELEC_CONS_RATE:
                        fold_shortest_path = nx.shortest_path(g, source=source_node, target=target_node, weight='weight')
                        fold_path[(source_node, target_node)] = fold_shortest_path
                        key_edges.append((source_node, target_node, modified_dis))
        mod_g.add_weighted_edges_from(key_edges)
        mod_loc = dict(map(lambda x: (x[1], (x[2], x[3])), key_nodes.itertuples()))
        # nx.draw(mod_g, mod_loc)
        # plt.show()
        return g, loc, mod_g, mod_loc, fold_path

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
        e_var_dict = m.addVars(g.nodes, vtype=GRB.CONTINUOUS, name='e', ub=100, lb=0)  # ELEC_MIN - 100
        s_var_dict = m.addVars(g.nodes, vtype=GRB.CONTINUOUS, name='s', ub=100, lb=0)

        # add objective function
        node_with_charger = [node[0] for node in g.nodes.data() if node[1]['charger'] == 1]
        price_coeff = tupledict(
            map(lambda node: (node[0], node[1]['price'] + VALUE_OF_TIME / CHARGING_RATE), g.nodes.data()))
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
                m.addConstr(x_var_dict.sum('*', node) * ELEC_CAPACITY >= s_var_dict[node], name='')

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
                # m.addConstr(e_var_dict[node] >= -(s_var_dict[node] - ELEC_MIN), name='')
            else:
                m.addConstr(e_var_dict[node] == 0, name='')

        m.update()

        print('约束数量%d' % len(m.getConstrs()))
        print('变量数量%d' % len(m.getVars()))
        m.optimize()
        if m.status == GRB.OPTIMAL:
            self.route = self._get_shortest_path(x_var_dict)
            # for edge in g.edges:
            #     print(edge, end='    ')
            #     print(x_var_dict[edge])
            charging = {}
            for node in g.nodes:
                if s_var_dict[node].X > 0.1:
                    charging[node] = e_var_dict[node].X
            print('total runtime of Gurobi Solver : %2f s' % m.Runtime)
            print('optimal route is {0}'.format(self.route))
            print('minimum cost of travelling is {0:d}'.format(int(m.ObjVal)))
            print('charging amount in each node: {0}'.format(charging))
            self.network_display()
        elif m.Status == GRB.Status.INFEASIBLE:
            raise Exception('master problem infeasible')
        else:
            raise Exception('sth went wrong')

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
                m.addConstr(x_var_dict.sum('*', node) * ELEC_CAPACITY >= s_var_dict[node], name='')

                # equation of SOC
                # expr_comp = []
                # for i, j, d in g.in_edges(node, data='weight'):
                #     expr_comp.append(x_var_dict[i, j] * quicksum([s_var_dict[i], - ELEC_CONS_RATE * d, q_var_dict[i]]))
                # m.addConstr(quicksum(expr_comp) == s_var_dict[node], name='soc in node %d' % node)

                # linearize the constraints
                for i, j, d in g.in_edges(node, data='weight'):
                    m.addConstr(s_var_dict[j] - s_var_dict[i] <= q_var_dict[i] - ELEC_CONS_RATE * d + BIG_M * (
                            1 - x_var_dict[(i, j)]), name='')
                    m.addConstr(s_var_dict[j] - s_var_dict[i] >= q_var_dict[i] - ELEC_CONS_RATE * d - BIG_M * (
                            1 - x_var_dict[(i, j)]), name='')

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

    @prompt_display
    def solve_by_benders_decomposition(self, ori_graph=True):
        # self.network_display()
        bddc = Bender_Decomposition(self.mod_G)
        route, cost, charging = bddc.master_problem()
        if ori_graph:
            unfold_route = []
            for path in route:
                unfold_path = self.fold_path[path]
                for index in range(len(unfold_path) - 1):
                    unfold_route.append((unfold_path[index], unfold_path[index + 1]))
            print(unfold_route)
            self.route = unfold_route
            self.network_display(modified=False)
            pass
        else:
            self.route = route
            self.network_display(modified=True)

    def network_display(self, mode='2dim', modified=False):
        if not modified:
            local_g = self.G
            local_loc = self.loc
        else:
            local_g = self.mod_G
            local_loc = self.mod_loc
        if mode == '2dim':
            color = {
                -1: '#0000C6',
                0: '#6C6C6C',
                1: '#000000',
                2: '#FF0000'
            }
            node_color = [color[para['charger']] for _, para in local_g.nodes.data()]
            edge_color = ['#FF0000' if edge in self.route else '#000000' for edge in local_g.edges]
            edge_width = [2.5 if edge in self.route else 1.0 for edge in local_g.edges]
            nx.draw(local_g, pos=local_loc, node_size=50, node_color=node_color,
                    edge_color=edge_color)  # edge_width=edge_width,
            # nx.draw_networkx_edges(self.G, edge_color=)
            plt.show()
        if self.route is None:
            pass


class Bender_Decomposition:
    def __init__(self, graph: nx.DiGraph):
        self.G = graph
        self.MP = Model()
        self.SP = Model()

    def master_problem(self):
        # TODO：MP 由最短路算法生成初始路径
        mp = self.MP
        mp.setParam('OutputFlag', 0)  # 不打印日志
        g = self.G
        init_node_of_path = nx.shortest_path(self.G, source=0, target=90, weight='weight')
        init_path = []
        for i in range(len(init_node_of_path) - 1):
            init_path.append((init_node_of_path[i], init_node_of_path[i + 1]))

        print('initial path is {0}'.format(init_path))

        # master problem construction
        # variable
        x_var_dict = mp.addVars(g.edges, vtype=GRB.BINARY, name='x')
        row_var = mp.addVar(vtype=GRB.CONTINUOUS, obj=1, name='row', column=None)

        # objective function
        dis_coeff = tupledict(map(lambda edge: ((edge[0], edge[1]), edge[2]['weight'] * COST_PER_DIS), g.edges.data()))
        mp.setObjective(x_var_dict.prod(dis_coeff) + row_var, GRB.MINIMIZE)

        # constraint
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
            mp.addConstr(flow_in_node_cstr == right_hand_side_value, name='flow in node %d' % node)

        plane_cutting_constraint: List[LinExpr] = []
        last_dual_variable = None
        total_runtime = 0  # total run time of Bender's decomposition
        # iteration
        while True:
            mp.update()
            mp.optimize()
            total_runtime = total_runtime + mp.Runtime

            # print('row value' + str(row_var.X))
            # for edge in self.G.edges:
            #     if x_var_dict[edge].X > 0.99:
            #         print(edge)
            # print(mp.ObjVal)

            if mp.Status == GRB.Status.OPTIMAL:
                # solve sub problem
                dual_variable, status, sp_runtime = self.sub_problem(x_var_dict)
                total_runtime = total_runtime + sp_runtime

                # terminate iteration only if the dual variable is exactly the same as the last iteration
                if last_dual_variable:
                    flag = True
                    for dual_index in range(len(dual_variable)):
                        if dual_variable[dual_index] != last_dual_variable[dual_index]:
                            flag = False
                            break
                    if flag:
                        break
                last_dual_variable = dual_variable

                if status >= 0:
                    #  optimality cut
                    if status == 0:
                        dual_iter = iter(dual_variable)
                        big_constraint = LinExpr()
                        for node, para in g.nodes.data():
                            if para['charger'] >= 0:
                                try:
                                    # lower bound of SOC
                                    # print('1:' + str(dual_variable[var]))
                                    big_constraint = big_constraint + x_var_dict.sum('*', node) * ELEC_MIN * \
                                                     next(dual_iter)

                                    # print('2:' + str(dual_variable[var]))
                                    big_constraint = big_constraint + x_var_dict.sum('*', node) * ELEC_CAPACITY * \
                                                     next(dual_iter)

                                    # equation of SOC
                                    for i, j, d in g.in_edges(node, data='weight'):
                                        # var = next(dual_index)
                                        # print('3:' + str(dual_variable[var]))

                                        # print(dual_variable[var] * (BIG_M * (
                                        #         1 - x_var_dict[(i, j)]) - ELEC_CONS_RATE * d))
                                        # print(dual_variable[var])

                                        big_constraint = big_constraint + next(dual_iter) * (BIG_M * (
                                                1 - x_var_dict[(i, j)]) - ELEC_CONS_RATE * d)

                                        # var = next(dual_index)
                                        # print('4:' + str(dual_variable[var]))
                                        big_constraint = big_constraint + next(dual_iter) * (-BIG_M * (
                                                1 - x_var_dict[(i, j)]) - ELEC_CONS_RATE * d)
                                except StopIteration:
                                    print('index overflow')
                        mp.addConstr(big_constraint <= row_var, name='optimality cut')
                    # feasibility cut
                    else:
                        boundary_value = 0
                        for shadow_price in dual_variable:
                            plane_cutting_constraint.append()
            elif mp.Status == GRB.Status.INFEASIBLE:
                raise Exception('master problem infeasible')
            else:
                raise Exception('sth went wrong')

        # print('row value' + str(row_var.X))
        route = []
        for edge in self.G.edges:
            if x_var_dict[edge].X > 0.99:
                route.append(edge)
        print('total runtime of Bender\'s Composition Function : %2f s' % total_runtime)

        sp_obj, charging_output = self.sub_problem(x_var_dict, obj_output=True)
        minimum_obj_value = mp.ObjVal - row_var.X + sp_obj
        print('optimal route is {0}'.format(route))
        print('minimum cost of travelling is {0:d}'.format(int(minimum_obj_value)))
        print('charging amount in each node: {0}'.format(charging_output))
        return route, minimum_obj_value, charging_output

    def sub_problem(self, x_var_dict: tupledict, obj_output=False):
        g = self.G
        sp = Model()
        # sp.Params.InfUnbdInfo = 1  # 设置后能输出极方向
        sp.setParam('OutputFlag', 0)  # 不打印日志
        e_var_dict = sp.addVars(g.nodes, vtype=GRB.CONTINUOUS, name='e', ub=100, lb=0)
        s_var_dict = sp.addVars(g.nodes, vtype=GRB.CONTINUOUS, name='s', ub=100, lb=0)

        # add objective function
        node_with_charger = [node[0] for node in g.nodes.data() if node[1]['charger'] == 1]
        price_coeff = tupledict(
            map(lambda node: (node[0], node[1]['price'] + VALUE_OF_TIME / CHARGING_RATE), g.nodes.data()))
        sp.setObjective(e_var_dict.prod(price_coeff, node_with_charger), GRB.MINIMIZE)

        # add constraints
        constrs: List[Constr] = []
        for node, para in g.nodes.data():
            if para['charger'] >= 0:
                node_indicator = sum([x_var_dict[edge].X for edge in g.in_edges(node)])
                # lower bound of SOC
                constrs.append(
                    sp.addConstr(node_indicator * ELEC_MIN <= s_var_dict[node], name='minimum soc in node %d' % node))
                constrs.append(sp.addConstr(node_indicator * ELEC_CAPACITY >= s_var_dict[node], name=''))

                # equation of SOC
                for i, j, d in g.in_edges(node, data='weight'):
                    flow_in_edge = 1 if x_var_dict[(i, j)].X > 0.99 else 0
                    constrs.append(
                        sp.addConstr(s_var_dict[j] - s_var_dict[i] - e_var_dict[i] <= - ELEC_CONS_RATE * d + BIG_M * (
                                1 - flow_in_edge), name=''))
                    constrs.append(
                        sp.addConstr(s_var_dict[j] - s_var_dict[i] - e_var_dict[i] >= - ELEC_CONS_RATE * d - BIG_M * (
                                1 - flow_in_edge), name=''))

            else:
                # origin node power constraint
                sp.addConstr(s_var_dict[node] == ELEC_CAPACITY, name='')

            # charging constraint
            if para['charger'] == 1:
                sp.addConstr(e_var_dict[node] <= ELEC_CAPACITY - s_var_dict[node], name='')
            else:
                sp.addConstr(e_var_dict[node] == 0, name='')

        sp.update()
        sp.optimize()

        # add benders_cut
        status = -1
        if sp.status == GRB.Status.OPTIMAL:
            if not obj_output:
                status = 0
                dual_variable = []
                for constr in constrs:
                    dual_variable.append(constr.Pi)
                return dual_variable, status, sp.Runtime
            else:
                charging_output = {}
                for node in self.G.nodes:
                    if s_var_dict[node].X > 0.1:
                        charging_output[node] = e_var_dict[node].X
                return sp.ObjVal, charging_output
        elif sp.status == GRB.Status.INFEASIBLE:
            # # not available for discontinuous model
            # raise Exception('MP is infeasible')
            status = 1
            extreme_dir = []
            raise Exception('opps')
            for constr in constrs:
                extreme_dir.append(constr.FarkasDual)
            return extreme_dir, status, sp.Runtime
        else:
            raise Exception('sth went wrong')


# def sub_problem(self, x_var_dict: tupledict):
#     g = self.G
#     sp = Model()
#     sp.Params.InfUnbdInfo = 1  # 设置后能输出极方向
#     q_var_dict = sp.addVars(g.nodes, vtype=GRB.CONTINUOUS, name='q', ub=100, lb=0)
#     e_var_dict = sp.addVars(g.nodes, vtype=GRB.CONTINUOUS, name='e', ub=100, lb=0)
#     r_var_dict = sp.addVars(g.nodes, vtype=GRB.BINARY, name='r')
#     s_var_dict = sp.addVars(g.nodes, vtype=GRB.CONTINUOUS, name='s', ub=100, lb=0)
#
#     # add objective function
#     node_with_charger = [node[0] for node in g.nodes.data() if node[1]['charger'] == 1]
#     price_coeff = tupledict(
#         map(lambda node: (node[0], node[1]['price'] + VALUE_OF_TIME / CHARGING_RATE), g.nodes.data()))
#     obj = e_var_dict.prod(price_coeff, node_with_charger) + REPLACE_FEE * r_var_dict.sum('*')
#     sp.setObjective(obj, GRB.MINIMIZE)
#
#     # add constraints
#     constrs: List[Constr] = []
#     for node, para in g.nodes.data():
#         if para['charger'] >= 0:
#             node_indicator = sum([x_var_dict[edge].Start for edge in g.in_edges(node)])
#             # lower bound of SOC
#             constrs.append(
#                 sp.addConstr(node_indicator * ELEC_MIN <= s_var_dict[node], name='minimum soc in node %d' % node))
#             constrs.append(sp.addConstr(node_indicator * BIG_M >= s_var_dict[node], name=''))
#
#             # equation of SOC
#             for i, j, d in g.in_edges(node, data='weight'):
#                 if x_var_dict[(i, j)].Start > 0.99:
#                     constrs.append(sp.addConstr(
#                         quicksum([s_var_dict[i], - ELEC_CONS_RATE * d, q_var_dict[i]]) == s_var_dict[node],
#                         name='soc in node %d' % node))
#         else:
#             # origin node power constraint
#             constrs.append(sp.addConstr(s_var_dict[node] == ELEC_CAPACITY, name=''))
#
#         # charging constraint
#         if para['charger'] == 1:
#             constrs.append(sp.addConstr(q_var_dict[node] <= ELEC_CAPACITY - s_var_dict[node], name=''))
#
#             # replacement indicator
#             constrs.append(
#                 sp.addConstr(q_var_dict[node] >= ELEC_CAPACITY - s_var_dict[node] - BIG_M * (1 - r_var_dict[node]),
#                              name=''))
#
#             # charging fee
#             constrs.append(sp.addConstr(e_var_dict[node] <= BIG_M * (1 - r_var_dict[node]), name=''))
#             constrs.append(sp.addConstr(e_var_dict[node] >= q_var_dict[node] - BIG_M * r_var_dict[node], name=''))
#         else:
#             constrs.append(sp.addConstr(q_var_dict[node] == 0, name=''))
#
#     sp.update()
#     print('约束数量%d' % len(sp.getConstrs()))
#     # print(constrs)
#     # print('变量数量%d' % len(sp.getVars()))
#     sp.optimize()
#
#     # add benders_cut
#     if sp.status == GRB.Status.OPTIMAL:
#         # print(constrs[2].ConstrName)
#         # print(constrs[2].getAttr('Pi'))
#         for constr in constrs:
#             print(constr.getAttr('ConstrName'))
#         pass
#     elif sp.status == GRB.Status.INFEASIBLE:
#         # not available for discontinuous model
#         raise Exception('MP is infeasible')
#         for constr in constrs:
#             print(type(constr))
#             print(constr.getAttr('FarkasDual'))
#     else:
#         raise Exception('sth went wrong')


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
    network.solve_single_by_gurobi()
    network.solve_by_benders_decomposition()
    exit()
    network.solve_alternative_by_gurobi()
    network.network_display()
