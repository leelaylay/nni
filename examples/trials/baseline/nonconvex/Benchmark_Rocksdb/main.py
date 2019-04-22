#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import logging
import nni
from rocksdb import bench


# set the logger format
log_format = "%(asctime)s %(message)s"
logging.basicConfig(
    filename="Metis-RocksDB.log",
    filemode="a",
    level=logging.DEBUG,
    format=log_format,
    datefmt="%m/%d %I:%M:%S %p",
)

LOG = logging.getLogger('RocksDB-Benchmark')


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
        'disable_wal': 'true'
    }
    return params


def add_offset(PARAMS):
    PARAMS['max_background_compactions'] += 1
    PARAMS['block_size'] += 1024
    PARAMS['write_buffer_size'] += 1048576
    PARAMS['max_write_buffer_number'] += 1
    PARAMS['min_write_buffer_number_to_merge'] += 1
    PARAMS['level0_file_num_compaction_trigger'] += 1
    PARAMS['level0_slowdown_writes_trigger'] += 1
    PARAMS['level0_stop_writes_trigger'] += 1
    PARAMS['cache_size'] += 1048576
    # PARAMS['compaction_readahead_size']+=0
    # PARAMS['new_table_reader_for_compaction_inputs']+=0

    return PARAMS


if __name__ == "__main__":
    try:
        # get parameters from tuner
        LOG.debug("Start Generating Parameter")
        RECEIVED_PARAMS = nni.get_next_parameter()
        LOG.debug("End Generating Parameter")

        PARAMS = get_default_parameters()
        PARAMS.update(RECEIVED_PARAMS)
        PARAMS = add_offset(PARAMS)

        result = bench(program='./db_bench', **PARAMS)
        ops_sec = result['fillrandom']
        LOG.debug("ops_sec:", ops_sec)

        nni.report_final_result(ops_sec)
    except Exception as exception:
        LOG.exception(exception)
        raise

