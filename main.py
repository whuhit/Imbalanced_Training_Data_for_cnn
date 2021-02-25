from configs.cifar_dla.dla_config import config
from agents.dla_cifar import DLAAgent


def main(config):
    cfg = vars(config)
    for i in range(1, 12):
        config_new = {
            "log_name": f"DLAAgent_simple_net_{i}.log",
            "train_file": f"assets/data/dla_cifar/dist_default/Dist_{i}.txt",
            "checkpoint_name": f"checkpoint-{i}.pth",
        }
        cfg.update(config_new)
        agent = DLAAgent(config)
        agent.run()


if __name__ == '__main__':
    main(config)
