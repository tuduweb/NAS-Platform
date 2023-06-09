import numpy as np
import copy
import os
from compute.file import get_algo_local_dir
from comm.log import Log
from comm.utils import GPUFitness
from algs.MO_evocnn_runner.utils import Utils
from algs.MO_evocnn_runner.genetic.statusupdatetool import StatusUpdateTool
from algs.MO_evocnn_runner.genetic.population import Population
from algs.MO_evocnn_runner.genetic.evaluate import FitnessEvaluate
from algs.MO_evocnn_runner.genetic.crossover_and_mutation import CrossoverAndMutation
from algs.MO_evocnn_runner.genetic.selection_operator import Selection
from compute.file import exec_cmd_remote, get_local_path, get_algo_name

import time
import pandas as pd

class EvolveCNN(object):
    def __init__(self, params):
        self.params = params
        self.pops = None
        self.pop_size = params['pop_size']
        self.M = 3  #the number of targets is 2
        self.lamda = np.zeros((self.pop_size, self.M))

        __k = 8 # vector_size -> 42 # self.pop_size // 2 # 子问题数量, 通常与pop_size相同
        _vector = []
        for i in range(0, __k):
            for j in range(0, __k):
                for k in range(0, __k):
                    if i + j + k == __k:
                        _vector.append([i, j, k])

        _vector_size = len(_vector)

        if _vector_size != self.pop_size:
            print("_vector_size and self.pop_size mismatch")
            exit(0)

        # for i in range(self.pop_size):
        #     self.lamda[i][0] = i / self.pop_size
        #     self.lamda[i][1] = (self.pop_size - i) / self.pop_size

        # 生成权重向量, 权重向量的数目需要和个体数目对应起来.. 不然后面会报错
        self.lamda = np.zeros((_vector_size, self.M))

        for idx, values in enumerate(_vector):
            i, k, j = values
            self.lamda[idx][0] = i / __k
            self.lamda[idx][1] = k / __k
            self.lamda[idx][2] = j / __k
        
        # 每一个权重向量的邻居 通常取 $\sqrt{N}$ 或 $N/10$。?
        self.T = int(self.pop_size / 5)
        # 每两个权重向量之间的欧几里得距离
        self.B = np.zeros((self.pop_size, self.pop_size))
        self.EP = []
        for i in range(self.pop_size):
            #x1, y1, z1 = _vector[i]
            for j in range(self.pop_size):
                #x2, y2, z2 = _vector[j]
                self.B[i][j] = np.linalg.norm(self.lamda[i, :] - self.lamda[j, :])
                # self.B[i][j] = np.linalg.norm(self.lamda[i, :] - self.lamda[j, :]);
            self.B[i, :] = np.argsort(self.B[i, :])

        # for i in range(_vector_size):
        #     for j in range(_vector_size):
        #         self.B[i][j] = np.linalg.norm(self.lamda[i, :] - self.lamda[j, :]);
        #     self.B[i, :] = np.argsort(self.B[i, :])


        self.z = np.zeros(self.M)
        for i in range(self.M):
            self.z[i] = 100
        self.z[self.M - 1] = 1000000

    def dominate(self, x, y):
        lte = 0
        lt = 0
        gte = 0
        gt = 0
        eq = 0
        for i in range(self.M):
            if x[i] <= y[i]:
                lte = lte + 1
            if x[i] < y[i]:
                lt = lt + 1
            if x[i] >= y[i]:
                gte = gte + 1
            if x[i] > y[i]:
                gt = gt + 1
            if x[i] == y[i]:
                eq = eq + 1
        if lte == self.M and lt > 0:
            return 1
        elif gte == self.M and gt > 0:
            return -1
        elif eq == self.M:
            return -2
        else:
            return 0

    # 以不同的比例相乘, 再取最大的
    def gte(self, f, lamda, z):
        return max(lamda * abs(f - z))
    
    def timeCostSmaller(self, timecost):
        # 10ms 0.2
        return timecost * 0.02

    def lossMeanSmaller(self, lossMean):
        # 0.1倍缩放
        return lossMean * 0.1

    def initialize_population(self):
        StatusUpdateTool.begin_evolution()
        pops = Population(self.params, 0)
        pops.initialize(Log)
        self.pops = pops
        Utils.save_population_at_begin(str(pops), 0)

    def modify_EP(self, indi):
        flag = 1
        j = 0
        candidate = [indi.error_mean, indi.loss_mean, self.timeCostSmaller(indi.timecost)]
        while j < len(self.EP):
            if j >= len(self.EP):
                break
            r = self.dominate(candidate, self.EP[j][0:self.M])
            if -2 == r:
                return -1
            if 1 == r:
                del self.EP[j]
                j -= 1
            elif -1 == r:
                flag = 0
            j += 1
        if flag == 1:
            candidate.append(indi.id)
            self.EP.append(candidate)
        return 0

    def fitness_evaluate(self, isFirst):
        fitness = FitnessEvaluate(self.pops.individuals, Log)
        fitness.generate_to_python_file()

        isCalTimeCost = True
        # 评价

        top_dir = get_local_path()
        algo_name = get_algo_name()

        runtime_dir = os.path.join(top_dir, 'runtime', algo_name)
        _sourcepath = os.path.realpath(os.path.join(runtime_dir, "scripts"))
        _runnerResourcePath = os.path.realpath(os.path.join(runtime_dir, "runner"))
        _summaryResultPath = os.path.realpath(os.path.join(runtime_dir, "summary"))

        """
        summary: 运行一次后的统计
        runnerLog: 执行器的时间消耗数据, 但是没有用到hash, 需要改进
        runnerMap: name -> uuid的结构
        """
        _summaryResultSavedUri = os.path.realpath(os.path.join(_summaryResultPath, "summary-gen-%05d.txt" % self.pops.gen_no))
        _runnerLogSavedUri = os.path.realpath(os.path.join(_runnerResourcePath, "runner-%05d.txt" % self.pops.gen_no))
        _runnerMapSavedUri = os.path.realpath(os.path.join(_runnerResourcePath, "map-%05d.txt" % self.pops.gen_no))


        """
        构造indi.id -> indi.uuid()[0] 对应表
        """
        indi_map = {}
        _str = []
        for indi in self.pops.individuals:
            indi_map[indi.id] = indi.uuid()[0]
            _str.append("%s=%s" % (indi.id, indi.uuid()[0]))

        Utils.write_to_file('\n'.join(_str), _runnerMapSavedUri)

        for key, value in indi_map.items():
            print(key, value)

        if isCalTimeCost:

            """
            try load from cache
            """
            cache_file_name = '%s/cache.txt' % (os.path.join(get_algo_local_dir(), 'runner'))
            
            # 尝试从以往已经运行的文件中读取..
            #if os.path.exists(_runnerLogSavedUri):

            """
            需要判断是否有该轮次的Runner缓存..
            如果没有的话就需要将该轮+需要比较的indi一起跑一次时间benchmark测试
            indi.uuid() = xxx.ncnn.param
            """
            _hash = "123456789helloworld"
            skipCheckHash = False

            _startTime = time.time()
            if not os.path.exists(_runnerLogSavedUri):
                _hash = "h%d" % int(time.time())
                _runner_py = "~/onebinary/ML-NAS/experiment/nas-evocnn-autodeploy/runner.py"
                _exec_cmd = "/usr/bin/python3 " + _runner_py + " --source=path --sourcepath=%s --machine=pi --hash=%s --gen=%d --saveuri=%s --mapuri=%s" % (_sourcepath, _hash, self.pops.gen_no, _runnerLogSavedUri, _runnerMapSavedUri)
                #_stdout, _stderr = exec_cmd_remote(_exec_cmd, need_response=False)
                Log.info("%s" % _exec_cmd)

                ret = os.system(_exec_cmd)

                _waitCnt = 0
                fileExist = False

                while not os.path.exists(_runnerLogSavedUri):
                    Log.info("wait for runner log, cnt %d" % _waitCnt)
                    time.sleep(5)
                    _waitCnt += 1

                    if _waitCnt > 100:
                        Log.info("error : wait for runner log, cnt %d" % _waitCnt)

            else:
                # 有缓存, 临时skip的情况
                skipCheckHash = True

            _okFlag = False

            while not _okFlag:
                f = open(_runnerLogSavedUri, 'r')
                lines = f.readlines()
                f.close()
                for line in lines:
                    if len(line.strip()) > 0:
                        if (skipCheckHash and "=" in line) or _hash in line:
                            line = line.strip().split('=')
                            datas = eval(line[1].strip())

                            for indi in self.pops.individuals:
                                if indi.id in datas.keys():
                                    indi.timecost = datas[indi.id]
                                else:
                                    indi.timecost = 999999.0 # + 1.0

                            _okFlag = True
                            break

                if not skipCheckHash:
                    Log.info("runner gen %d: wait %fs" % (self.pops.gen_no, (time.time() - _startTime) * 1))
                    time.sleep(5)

            _endTime = time.time()
            Log.info("runner gen %d: timecost %fs" % (self.pops.gen_no, (_endTime - _startTime) * 1))

            """
            理论上 在这里已经:
                1. 获取到了缓存数据中的benchmark基准
                2. (or) 跑了一次benchmark测试 得到了时间数据..这个数据是可以保存的, 因为跟gen_begin是相对应的..
            TODO: .ncnn.param文件是可以根据模型uuid复用的, 在程序中, 模型的uuid根据结构生成的字符串, 取加密算法获得一串uuid
            """

            for indi in self.pops.individuals:
                timecost = indi.timecost
                Log.info("indi %s timecost %f" % (indi.id, timecost))

        else:
            """
            不关心时间消耗
            """
            pass


        # for indi in self.pops.individuals:
        #     if indi.acc_mean == -1:
        #         indi.acc_mean = np.random.random()
        fitness.evaluate()
        time.sleep(3)
        fitness_map = GPUFitness.read()
        # for indi in self.pops.individuals:
        #     if indi.error_mean == -1 and indi.id in fitness_map.keys():
        #         indi.error_mean, indi.loss_mean = fitness_map[indi.id][0], fitness_map[indi.id][1]

        for indi in self.pops.individuals:
            _retryCnt = 0
            brkFlag = False
            if indi.error_mean == -1: #modify_timecost version
                if indi.timecost >= 1000:
                    Log.info("%s time cost[%.5f] too much, skip." % (indi.id, indi.timecost))
                    continue
        
                while indi.id not in fitness_map.keys():
                    Log.info('Fitness evaludate cant found: indi.id[%s] not in fitness_map(size = %d), sleep and retry %d' % (indi.id, len(fitness_map), _retryCnt))
                    time.sleep(10)
                    fitness_map = GPUFitness.read()
                    _retryCnt += 1

                    if _retryCnt % 10 == 0:
                        txt = Utils.load_train_log(indi.id)
                        if len(txt) > 0 and "RuntimeError" in "".join(txt[-5:-1]):
                            brkFlag = True
                            break

                    # 或者直接读取indi里的最后一行
                    if _retryCnt >= 100:
                        if indi.id not in fitness_map.keys():
                            brkFlag = True
                            break

                if not brkFlag:
                    # 从results.txt中读取.. results.txt 是从remote的training中获取, 通过redis传递
                    indi.error_mean, indi.loss_mean, indi.paramnum = fitness_map[indi.id][0], fitness_map[indi.id][1], fitness_map[indi.id][2]
                else:
                    Log.info('Fitness evaludate skip: indi.id[%s] not in fitness_map(size = %d), after retry %d' % (indi.id, len(fitness_map), _retryCnt))

        """
        生成"临时"可视化文件, 体现出直观的数据..
        """
        viewerList = []
        for indi in self.pops.individuals:
            viewerLine = {
                "name": indi.id,
                "uuid": indi.uuid()[0], # 跟模型结构高度相关
                "error_mean": indi.error_mean,
                "loss_mean": indi.loss_mean,
                "time_cost": indi.timecost,
                "complexity": indi.complexity, # 复杂度
                "param_num": indi.paramnum # 参数量
            }
            viewerList.append(viewerLine)

        form_header = viewerList[0].keys()
        df = pd.DataFrame(columns=form_header)

        for idx, item in enumerate(viewerList):
            df.loc[idx] = item.values()

        df.sort_values("name", inplace=True, ascending=True)
        # save to
        df.to_csv(_summaryResultSavedUri, index=False)
        
        Log.info('Summary gen [%d] to %s' % (self.pops.gen_no, _summaryResultSavedUri))


        """
        决策过程
        """
        for indi in self.pops.individuals:
            if indi.error_mean != -1:
                if self.z[0] > indi.error_mean:
                    self.z[0] = indi.error_mean
                if self.z[1] > indi.loss_mean:
                    self.z[1] = indi.loss_mean
                if self.z[2] > self.timeCostSmaller(indi.timecost):
                    self.z[2] = self.timeCostSmaller(indi.timecost)
                if -1 == self.modify_EP(indi):
                    Log.info('%s has duplicate' % (indi.id))
        Utils.save_EP_after_evaluation(str(self.EP), self.pops.gen_no)
        if isFirst == False:
            self.pops.offsprings = copy.deepcopy(self.pops.individuals)
            self.pops.individuals = copy.deepcopy(self.pops.parent_individuals)

    def crossover_and_mutation(self):
        params = {}
        params['crossover_eta'] = StatusUpdateTool.get_crossover_eta()
        params['mutation_eta'] = StatusUpdateTool.get_mutation_eta()
        params['acc_mean_threshold'] = StatusUpdateTool.get_acc_mean_threshold()
        params['complexity_threshold'] = StatusUpdateTool.get_complexity_threshold()
        cm = CrossoverAndMutation(self.params['genetic_prob'][0], self.params['genetic_prob'][1], Log,
                                  self.pops.individuals, self.B, self.T, self.pops.gen_no, params)
        offspring = cm.process()
        self.pops.parent_individuals = copy.deepcopy(self.pops.individuals)
        self.pops.individuals = copy.deepcopy(offspring)

    def environment_selection(self):
        v_list = []
        indi_list = []
        _str = []
        # 在一次完整的..里, 由生成begin, 和交叉变异后子代构成
        """
        self.pops.offsprings = copy.deepcopy(self.pops.individuals) # 刚刚训练出炉的一带
        self.pops.individuals = copy.deepcopy(self.pops.parent_individuals) # 上一代
        """
        for indi in self.pops.individuals:
            indi_list.append(indi)
            v_list.append(indi.error_mean)
            _t_str = 'Indi-%s-%.5f-%.5f-%.5f-%s' % (indi.id, indi.error_mean, indi.loss_mean, indi.timecost, indi.uuid()[0])
            _str.append(_t_str)
        for indi in self.pops.offsprings:
            indi_list.append(indi)
            v_list.append(indi.error_mean)
            _t_str = 'Offs-%s-%.5f-%.5f-%.5f-%s' % (indi.id, indi.error_mean, indi.loss_mean, indi.timecost, indi.uuid()[0])
            _str.append(_t_str)

        """
        在 MOEA/D 中，通过将权重向量分配给每个个体，可以将多目标优化问题转化为单目标优化问题。
        具体地，对于每个个体，使用其对应的权重向量计算出其在目标空间中的投影值（也称为加权向量），
        然后将其与邻居个体的加权向量进行比较，选择其中最优的一个作为该个体的邻居。


        在MOEA/D算法中，选择邻居向量的过程被称为"neighboring selection"，
        其基本思想是从预定义的权重向量集合中选择一些与当前个体比较接近的权重向量，
        然后在这些权重向量对应的子问题上，选择一些比较优秀的邻居进行交叉和变异操作，得到新的解并更新当前个体的状态。
        """
        i = 0
        while i < len(self.pops.offsprings):
            # offsprings是刚得出训练结果的那一代
            indi = copy.deepcopy(self.pops.offsprings[i])
            # 在权重向量的邻居中 self.T 邻居个数
            for j in range(self.T):
                p = int(self.B[i, j])
                o = copy.deepcopy(self.pops.individuals[p])

                """
                在MOEA/D算法中GTE:Global Thresholding Elimination是一个用于选择邻居个体的阈值
                """
                # 生成的子代的循环 i value_fj子代
                # 原始, i的表现, 在其第j个邻居位置上的表现
                value_fj = self.gte([indi.error_mean, indi.loss_mean, self.timeCostSmaller(indi.timecost)], self.lamda[p, :], self.z) # 子问题p..
                # o. i的 第j个邻居. 在其位置上的表现
                value_p = self.gte([o.error_mean, o.loss_mean, self.timeCostSmaller(o.timecost)], self.lamda[p, :], self.z)

                # 然后将其与邻居个体的加权向量进行比较，选择其中最优的一个作为该个体的邻居。
                """
                具体来说，邻居向量的选择可以通过以下几种方法实现：
                Tchebycheff approach：选择在当前个体的参考向量中距离最小的向量作为邻居向量。
                Neighborhood search approach：从当前个体附近的权重向量中选择若干个向量作为邻居向量。
                Decomposition-based approach：根据子问题上个体之间的相似度选择邻居向量。
                通过选择合适的邻居向量，MOEA/D算法可以增加算法搜索空间的覆盖率，从而更好地探索目标函数的全局结构，同时也可以促进算法的多样性和收敛性。


                在多目标优化问题中，平衡不同目标函数之间的矛盾是非常关键的。MOEA/D算法采用的一种常见的平衡方法是通过权重向量来平衡不同目标函数之间的关系。

                具体来说，MOEA/D算法生成一组均匀分布的权重向量，这些权重向量在目标函数空间中均匀地分布。每个权重向量都对应一个目标函数组合，表示不同目标函数的权重。在算法的演化过程中，每个个体都会被分配到最接近它的权重向量所代表的目标函数组合。这样，每个个体都会被优化在其所属的目标函数组合下，从而实现了目标函数之间的平衡。

                此外，MOEA/D算法还采用了一种叫做“分解”的技术来平衡不同目标函数之间的关系。分解技术将多目标优化问题转化为多个单目标优化子问题，并通过分别求解这些子问题来获得整个问题的最优解。在MOEA/D算法中，每个子问题对应一个权重向量，每个个体都会被分配到最接近它的权重向量所代表的子问题中进行优化。这样，每个个体都会在多个子问题中参与优化，从而实现了目标函数之间的平衡。

                总之，MOEA/D算法通过权重向量和分解技术来平衡不同目标函数之间的关系，从而实现多目标优化问题的有效求解。
                """
                if value_fj < value_p:
                    self.pops.individuals[p] = copy.deepcopy(indi)
                # 最后，从每个个体的邻居中选择一个作为其父代，并使用交叉和变异操作来生成下一代个体。.. crossover_and_mutation
            i += 1


        # 通过上面, 改变了indi的顺序, 记录下来看看
        for indi in self.pops.individuals:
            indi_list.append(indi)
            # v_list.append(indi.error_mean)
            _t_str = 'Aftr-%s-%.5f-%.5f-%.5f-%s' % (indi.id, indi.error_mean, indi.loss_mean, indi.timecost, indi.uuid()[0])
            _str.append(_t_str)

        _file = '%s/ENVI_%05d.txt' % (os.path.join(get_algo_local_dir(), 'populations'), self.pops.gen_no)
        Utils.write_to_file('\n'.join(_str), _file)

        """Here, the population information should be updated, such as the gene no and then to the individual id"""
        next_gen_pops = Population(self.pops.params, self.pops.gen_no + 1)
        next_gen_pops.create_from_offspring(self.pops.individuals)
        self.pops = next_gen_pops
        for _, indi in enumerate(self.pops.individuals):
            _t_str = 'new -%s-%.5f-%.5f-%.5f-%s' % (indi.id, indi.error_mean, indi.loss_mean, indi.timecost, indi.uuid()[0])
            _str.append(_t_str)
        _file = '%s/ENVI_%05d.txt' % (os.path.join(get_algo_local_dir(), 'populations'), self.pops.gen_no - 1)
        Utils.write_to_file('\n'.join(_str), _file)

        Utils.save_population_at_begin(str(self.pops), self.pops.gen_no)

    def create_necessary_folders_and_init(self):
        sub_folders = [os.path.join(get_algo_local_dir(), v) for v in ['populations', 'log', 'scripts', 'runner', 'summary']]
        if not os.path.exists(get_algo_local_dir()):
            os.mkdir(get_algo_local_dir())
        for each_sub_folder in sub_folders:
            if not os.path.exists(each_sub_folder):
                os.mkdir(each_sub_folder)
        
        # 将配置项整合到一个文件里
        #from compute import config

        #get_global_ini_path()
        #get_train_ini_path()
        from comm.utils import PlatENASConfig
        g = PlatENASConfig('algorithm')
        algs_name = g.read_ini_file('run_algorithm')

        # global.ini, compute/gpu.ini, algs/xx/genetic/global.ini, train/train.ini
        configFiles = ['global.ini', 'compute/gpu.ini', 'train/train.ini', os.path.join('algs', algs_name, 'genetic/global.ini')]
        # configFiles = [os.path.realpath(os.path.join(os.path.dirname(os.path.dirname(__file__)), uri)) for uri in configFilesUri]

        config_lines = []
        for _file in configFiles:
            fileUri = os.path.realpath(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), _file))
            # 额外增加注释项
            with open(fileUri, 'r') as f:
                filelines = f.readlines()
                f.close()
                config_lines.append("; [%s] path: %s" % (_file, fileUri))
                config_lines.extend(filelines)


        Utils.write_to_file("".join(config_lines), os.path.join(get_algo_local_dir(), "merge_config.ini"))

        # db_ip = g.read_ini_file('log_server')
        # db_port = int(g.read_ini_file('pop_size'))

    def do_work(self, max_gen):
        # create the corresponding fold under runtime
        self.create_necessary_folders_and_init()

        # the step 1
        if StatusUpdateTool.is_evolution_running():
            Log.info('Initialize from existing population data')
            gen_no = Utils.get_newest_file_based_on_prefix('begin')
            if gen_no is not None:
                Log.info('Initialize from %d-th generation' % (gen_no))
                pops = Utils.load_population('begin', gen_no)
                self.pops = pops
                if gen_no > 0:
                    EP = Utils.load_EP('EP', gen_no - 1)
                    self.EP = EP
            else:
                raise ValueError('The running flag is set to be running, but there is no generated population stored')
        else:
            gen_no = 0
            Log.info('Initialize...')
            self.initialize_population()
        Log.info('EVOLVE[%d-gen]-Begin to evaluate the fitness' % (gen_no))
        self.fitness_evaluate(True)
        Log.info('EVOLVE[%d-gen]-Finish the evaluation' % (gen_no))

        for curr_gen in range(gen_no+1, max_gen+1):
            self.params['gen_no'] = curr_gen
            self.pops.gen_no = curr_gen
            # step 3
            Log.info('EVOLVE[%d-gen]-Begin to crossover and mutation' % (self.pops.gen_no))
            self.crossover_and_mutation()
            Log.info('EVOLVE[%d-gen]-Finish crossover and mutation' % (self.pops.gen_no))

            Log.info('EVOLVE[%d-gen]-Begin to evaluate the fitness' % (self.pops.gen_no))
            self.fitness_evaluate(False)
            Log.info('EVOLVE[%d-gen]-Finish the evaluation' % (self.pops.gen_no))

            self.environment_selection()
            Log.info('EVOLVE[%d-gen]-Finish the environment selection' % (
                    self.pops.gen_no - 1))  # in environment_selection, gen_no increase 1
        StatusUpdateTool.end_evolution()


class Run():
    def do(self):
        params = StatusUpdateTool.get_init_params()
        evoCNN = EvolveCNN(params)
        evoCNN.do_work(params['max_gen'])


if __name__ == '__main__':
    r = Run()
    r.do()
    # params = StatusUpdateTool.get_init_params()
    # evoCNN = EvolveCNN(params)
    # evoCNN.create_necessary_folders()
    # evoCNN.initialize_population()
    # evoCNN.pops = Utils.load_population('begin', 0)
    # evoCNN.fitness_evaluate()
    # evoCNN.crossover_and_mutation()
