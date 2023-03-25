import os, configparser
from compute.file import get_algo_local_dir
from compute.redis import RedisLog


class PlatENASConfig(object):

    def __init__(self, section):
        file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'global.ini')  # global.ini为配置文件
        self.config_file = file_path
        self.section = section

    # 从global.ini中读取配置
    def read_ini_file(self, key):
        config = configparser.ConfigParser()
        config.read(self.config_file)
        return config.get(self.section, key)


class GPUFitness():
    @classmethod
    def read(cls):
        file_name = '%s/populations/results.txt' % (get_algo_local_dir())
        if not os.path.exists(file_name):
            file = open(file_name, 'w')
            file.close()
        f = open(file_name, 'r')
        fitness_map = {}
        for line in f:
            if len(line.strip()) > 0:
                line = line.strip().split('=')

                datas = line[1].strip().split(',')

                fitness_map[line[0]] = []

                # for i in len(datas):
                #     datas[i]
                #     fitness_map[line[0]].append(float(datas[i]))

                fitness_map[line[0]].append(float(datas[0])) # error_mean
                fitness_map[line[0]].append(float(datas[1])) # loss_mean

                fitness_map[line[0]].append(int(datas[2])) # 参数量

        f.close()
        return fitness_map


class CacheToResultFile():
    @classmethod
    def do(cls, file_id, best_error, best_loss, num_para = 99999999999):
        logger = RedisLog(os.path.basename(file_id) + '.txt')
        logger.write_file('RESULTS', 'results.txt', '%s=%.5f,%.5f,%d\n' % (file_id, best_error, best_loss, num_para))

# class CacheToResultFile():
#     @classmethod
#     def do(cls, file_id, best_error):
#         logger = RedisLog(os.path.basename(file_id) + '.txt')
#         logger.write_file('RESULTS', 'results.txt', '%s=%.5f\n' % (file_id, best_error))

if __name__ == '__main__':
    GPUFitness.read()
