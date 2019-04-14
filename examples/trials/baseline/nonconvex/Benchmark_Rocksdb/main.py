#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import logging
import subprocess

import nni

LOG = logging.getLogger('rocksdb-benchmark')

def bench(program='./db_bench', **parameters):
    program = [program] if isinstance(program, str) else list(program)
    args = [f'--{k}={v}' for k, v in parameters.items()]
    process = subprocess.Popen(program + args, stdout=subprocess.PIPE)
    out, err = process.communicate()
    lines = out.decode('utf8').splitlines()
    i = 0
    while lines[i].strip() != '-' * 48:
        i += 1
    while not lines[i].startswith('DB path:'):
        i += 1
    i += 1
    result = {}
    for line in lines[i:]:
        key, _, value = line.partition(':')
        key = key.strip()
        value_right = value.find(' ops/sec')
        value_left = value.rfind(' ', 0, value_right)
        value = float(value[value_left:value_right])
        result[key] = value
    return result

def get_default_parameters():
    params = {
        "benchmarks":'fillrandom',
        'target_file_size_base': 134217728,
        'max_bytes_for_level_base': 67108864,
        'num_levels': 6,
        'compression_type': 'zstd',
        'sync': 'false',
        'max_background_jobs': 24,
        'open_files': -1,
        'bloom_bits': 10,
        'use_direct_reads': 'true',
        'use_direct_io_for_flush_and_compaction': 'true',
        'allow_concurrent_memtable_write': 'false',
        'enable_write_thread_adaptive_yield': 'true',
        'random_access_max_buffer_size': 0,
        'writable_file_max_buffer_size': 2097152,
        'disable_auto_compactions': 'false',
        'disable_wal': 'true',
        'nni_column_family': 'CrawlMetaCG',
    }
    return params


if __name__ == "__main__":
    try:
        # get parameters from tuner
        LOG.debug("Start Generating Parameter")
        RECEIVED_PARAMS = nni.get_next_parameter()
        LOG.debug("End Generating Parameter")

        PARAMS = get_default_parameters()
        PARAMS.update(RECEIVED_PARAMS)

        result = bench(program='./db_bench', **PARAMS)
        ops_sec = result['fillrandom']
        LOG.debug("ops_sec:", ops_sec)

        nni.report_final_result(ops_sec)
    except Exception as exception:
        LOG.exception(exception)
        raise
