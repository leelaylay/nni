import subprocess


def bench(program='./db_bench', **parameters):
    """for fillrandom benchmark and readrandom benchmark
    """
    program = [program] if isinstance(program, str) else list(program)
    # recover args
    args = [f'--{k}={v}' for k, v in parameters.items()]
    # subprocess communicate
    process = subprocess.Popen(program + args, stdout=subprocess.PIPE)
    out, err = process.communicate()
    # split into lines
    lines = out.decode('utf8').splitlines()

    match_lines = []
    for line in lines:
        # find the line with matched str
        if "ops/sec;" not in line:
            continue
        else:
            match_lines.append(line)
            break

    results = {}
    for line in match_lines:
        key, _, value = line.partition(':')
        key = key.strip()
        value = value.split("op")[1]
        results[key] = float(value)

    return results

# old version implement of bench
# def bench(program='./db_bench', **parameters):
#     program = [program] if isinstance(program, str) else list(program)
#     args = [f'--{k}={v}' for k, v in parameters.items()]
#     process = subprocess.Popen(program + args, stdout=subprocess.PIPE)
#     out, err = process.communicate()
#     lines = out.decode('utf8').splitlines()
#     i = 0
#     while lines[i].strip() != '-' * 48:
#         i += 1
#     while not lines[i].startswith('DB path:'):
#         i += 1
#     i += 1
#     result = {}
#     for line in lines[i:]:
#         key, _, value = line.partition(':')
#         key = key.strip()
#         value_right = value.find(' ops/sec')
#         value_left = value.rfind(' ', 0, value_right)
#         value = float(value[value_left:value_right])
#         result[key] = value
#     return result
