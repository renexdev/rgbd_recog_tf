import sys, os, shutil

result_dir = 'results'


def move_files(folder, dest):
    for i in os.listdir(folder):
        src = os.path.join(folder, i)
        shutil.move(src, dest)


if __name__ == '__main__':
    name = sys.argv[1]
    path = os.path.join(result_dir, name)
    ckpt_path = os.path.join(path, 'checkpoints')
    bestckpt_path = os.path.join(path, 'bestckpt')
    log_path = os.path.join(path, 'logs')
    score_path = os.path.join(path, 'score')
    summary_path = os.path.join(path, 'summary')

    if not os.path.exists(path):
        os.mkdir(path)
        os.mkdir(ckpt_path)
        os.mkdir(bestckpt_path)
        os.mkdir(log_path)
        os.mkdir(score_path)
        os.mkdir(summary_path)

    move_files('checkpoints', ckpt_path)
    move_files('bestckpt', bestckpt_path)
    move_files('logs', log_path)
    move_files('score', score_path)
    move_files('summary', summary_path)
