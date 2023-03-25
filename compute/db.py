import sqlite3
import os
import time
from compute.log import Log
from compute.config import AlgorithmConfig, ExecuteConfig
from compute.file import get_local_path


def get_db_name():
    return 'PlatENAS'


def get_db_path():
    file_path = os.path.join(get_local_path(), 'runtime', '%s.db' % (get_db_name()))
    return file_path


def init_db():
    Log.info('Initialize the database')
    try:
        conn = sqlite3.connect(get_db_path())
        rs = check_table(conn, 'gpu_list')
        if len(rs) == 0:
            Log.debug('Tables do not exist, creating them first')
            _create_table_sql = '''CREATE TABLE %s
                               (id  integer PRIMARY KEY autoincrement,
                               alg_name              VARCHAR(250),
                               worker                VARCHAR(250),		     
                               ssh_name              VARCHAR(250),
                               ssh_password          VARCHAR(250),
                               ssh_port              VARCHAR(250),
                               gpu_id                int(8),
                               status                int(8),
                               task_num              int(8),
                               remark                text,
                               time                  VARCHAR(100));'''

            Log.debug('Init the gpu_list table ...')
            conn.execute(_create_table_sql % ('gpu_list'))
            Log.debug('Init the gpu_arxiv_list table ...')
            conn.execute(_create_table_sql % ('gpu_arxiv_list'))

            # init the gpu_use table
            _create_table_sql = '''CREATE TABLE gpu_use
                                   (id integer PRIMARY KEY autoincrement,
                                   alg_name               VARCHAR(250),
                                   worker                 VARCHAR(250),
                                   gpu_id                 VARint(8),
                                   status                 VARCHAR(5),
                                   script_name            VARCHAR(250),
                                   time                   VARCHAR(100));'''
            Log.debug('Init the gpu_use table ...')
            conn.execute(_create_table_sql)

            conn.commit()
        else:
            Log.debug('Table gpu_list already exists')

        conn.close()
    except BaseException as e:
        Log.warn('Errors when initializing the database [%s]' % (str(e)))


def check_table(conn, table_name):
    sql = 'SELECT name FROM sqlite_master WHERE type=\'table\' AND name=\'%s\';' % (table_name)
    cu = conn.cursor()
    cu.execute(sql)
    rs = cu.fetchall()
    return rs


def add_info(gpu_info, info):
    '''
    Two tables will be operated, gpu_list and gpu_arxiv_list
    all the gpu infomation should be added to gpu_arxiv_list
    while only the available gpu list is added to gpu_list, because operating this table, a delete operation should be given
    '''
    conn = sqlite3.connect(get_db_path())
    alg_config = AlgorithmConfig()
    cu = conn.cursor()
    alg_name = alg_config.read_ini_file('name')
    time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))

    excute_config = ExecuteConfig()
    per_gpu_task_max = int(excute_config.read_ini_file('per_gpu_task_max'))

    sql_gpu_list_del = 'delete from gpu_list where alg_name = \'%s\'' % (alg_name)
    Log.debug('Execute sql: %s' % (sql_gpu_list_del))
    cu.execute(sql_gpu_list_del)

    sql_gpu_list_del2 = 'delete from gpu_arxiv_list where alg_name = \'%s\'' % (alg_name)
    Log.debug('Execute sql: %s' % (sql_gpu_list_del2))
    cu.execute(sql_gpu_list_del2)

    for each_one in info:

        worker = each_one['worker_ip']
        ssh_name = gpu_info[each_one['worker_ip']]['ssh_name']
        ssh_password = gpu_info[each_one['worker_ip']]['ssh_password']
        ssh_port = gpu_info[each_one['worker_ip']]['ssh_port']
        gpu_id = each_one['gpu_id']
        status = each_one['status']
        remark = each_one['remark']
        """
        根据compute平台读取的, 已经运行的数量, 跟允许最大运行的数量比较, 来决定是否加到"空闲"列表中
        这里有的问题是, 没有根据负载情况来合理判断添加, 而只是根据了剩余内存的情况盲目让status置0或者1
        task_cnt = 0, 2 can be used
        task_cnt = 1, 1 can be
        task_cnt = 2 cant
        """
        task_cnt = int(each_one['task_cnt'])
        task_num = task_cnt
        _per_max = per_gpu_task_max

        if status == 0:
            Log.info("GPU[{gpu_id}] runs task_num {task_cnt}, max {_per_max}"
                     .format(gpu_id = gpu_id, task_cnt = task_cnt, _per_max = _per_max))
    
            for i in range(_per_max - task_cnt):
                sql_gpu_list = 'INSERT INTO gpu_list(alg_name, worker, ssh_name, ssh_password, ssh_port, gpu_id, status, task_num, remark, time) values (\'%s\', \'%s\', \'%s\', \'%s\', \'%s\', %d, %d, %d, \'%s\', \'%s\')' % (
                    alg_name, worker, ssh_name, ssh_password, ssh_port, gpu_id, status, task_num, remark, time_str
                )
                Log.debug('Execute sql: %s' % (sql_gpu_list))
                cu.execute(sql_gpu_list)

        sql_gpu_archiv_list = 'INSERT INTO gpu_arxiv_list(alg_name, worker, ssh_name, ssh_password, ssh_port, gpu_id, status, task_num, remark, time) values (\'%s\', \'%s\', \'%s\', \'%s\', \'%s\', %d, %d, %d, \'%s\', \'%s\')' % (
            alg_name, worker, ssh_name, ssh_password, ssh_port, gpu_id, status, task_num, remark, time_str
        )

        Log.debug('Execute sql: %s' % (sql_gpu_archiv_list))
        cu.execute(sql_gpu_archiv_list)
    conn.commit()
    conn.close()

# 如果承载能力为2倍, 那么这个值可能为2倍
def get_available_gpus():
    alg_config = AlgorithmConfig()
    alg_name = alg_config.read_ini_file('name')
    sql = 'select id, worker as worker_ip, gpu_id, ssh_name, ssh_password, ssh_port, task_num from gpu_list where alg_name=\'%s\' order by task_num' % (
        alg_name)
    Log.debug('Execute sql: %s' % (sql))
    conn = sqlite3.connect(get_db_path())
    cu = conn.cursor()
    cu.execute(sql)
    rs = cu.fetchall()
    conn.close()
    return rs

def get_all_gpus_infos_from_database():
    alg_config = AlgorithmConfig()
    alg_name = alg_config.read_ini_file('name')
    # task_num: 关键字的num
    sql = 'select id, worker as worker_ip, gpu_id, ssh_name, ssh_password, ssh_port, task_num from gpu_arxiv_list where alg_name=\'%s\'' % (
        alg_name)
    Log.debug('Execute sql: %s' % (sql))
    conn = sqlite3.connect(get_db_path())
    cu = conn.cursor()
    cu.execute(sql)
    rs = cu.fetchall()
    conn.close()

    return rs

def confirmed_used_gpu(ids):
    sql = 'delete from gpu_list where id in (%s)' % (','.join(ids))
    Log.debug('Execute sql: %s' % (sql))
    conn = sqlite3.connect(get_db_path())
    cu = conn.cursor()
    cu.execute(sql)
    conn.commit()
    conn.close()

# 自己添加的新的
def confirmed_used_gpu_task(ids):
    sql = 'delete from gpu_list where id in (%s)' % (','.join(ids))
    Log.debug('Execute sql: %s' % (sql))
    conn = sqlite3.connect(get_db_path())
    cu = conn.cursor()
    cu.execute(sql)
    conn.commit()
    conn.close()
