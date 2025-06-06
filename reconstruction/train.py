from utils import *
from options import Options


def get_method(opt):
    print("=> Running Method: {}".format(opt.model['name']))
    if opt.model['name'] in ['ae', 'ae-grad', 'ae-spatial', 'ae-ssim', 'ae-l1', 'ae-perceptual']:
        return AEWorker(opt)
    elif opt.model['name'] in ['ae-qb', 'aeu-qb', 'aeu-perceptual-qb']:
        return AEU_QBWorker(opt)
    elif opt.model['name'] == 'memae':
        return MemAEWorker(opt)
    elif opt.model['name'] == 'aeu':
        return AEUWorker(opt)
    elif 'vae' in opt.model['name']:
        return VAEWorker(opt)
    elif opt.model['name'] == 'ceae':
        return CeAEWorker(opt)
    elif opt.model['name'] == 'ganomaly':
        return GanomalyWorker(opt)
    elif opt.model['name'] == 'constrained-ae':
        return ConstrainedAEWorker(opt)
    elif opt.model['name'] == 'dae':
        return DAEWorker(opt)
    else:
        raise Exception("Invalid model name: {}".format(opt.model['name']))


def main():
    opt = Options(isTrain=True)
    opt.parse()
    opt.save_options()

    worker = get_method(opt)
    worker.set_gpu_device()
    worker.set_seed()
    worker.set_network_loss()
    worker.set_optimizer()
    worker.set_dataloader()
    worker.set_logging()
    worker.run_train()


if __name__ == "__main__":
    main()
