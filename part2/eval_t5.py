import argparse

from t5_utils import load_model_from_checkpoint
from load_data import load_t5_data
from train_t5 import eval_epoch, test_inference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', type=str, default='experiment')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--test_batch_size', type=int, default=32)
    args = parser.parse_args()
    args.finetune = True

    _, dev_loader, test_loader = load_t5_data(args.batch_size, args.test_batch_size)
    model = load_model_from_checkpoint(args, best=True)
    model.eval()

    name = args.experiment_name
    dev_loss, f1, record_em, sql_em, error_rate = eval_epoch(
        args, model, dev_loader,
        'data/dev.sql', f'results/t5_ft_{name}_dev.sql',
        'records/ground_truth_dev.pkl', f'records/t5_ft_{name}_dev.pkl')
    print(f"Dev set results: Loss: {dev_loss}, Record F1: {f1}, Record EM: {record_em}, SQL EM: {sql_em}")
    print(f"Dev set results: {error_rate*100:.2f}% of the generated outputs led to SQL errors")

    test_inference(args, model, test_loader,
                   f'results/t5_ft_{name}_test.sql', f'records/t5_ft_{name}_test.pkl')


if __name__ == "__main__":
    main()
