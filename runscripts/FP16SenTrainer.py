from Basetrainer import ObjectDetectionPipeline
import argparse
from ast import literal_eval

def parse_args():
    def parse_band_list(band_str):
        """Parse a comma-separated string into a list of integers."""
        lista = literal_eval(band_str)
        if not isinstance(lista, list):
            assert isinstance(lista, int), 'The band list must be a list or a single integer'
            lista = [lista]
        return lista
    
    parser = argparse.ArgumentParser(description='Train a model for object detection')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--band', type=parse_band_list, help='The band list num to train(list)', default='[2,3,4,8]')
    parser.add_argument('--seed', type=int, default=54, help='Random seed')
    parser.add_argument('--learning_rate', type=float, default=0.0005, help='Learning rate')
    parser.add_argument('--max_epochs', type=int, default=130, help='Max epochs')
    parser.add_argument('--sensor', type=str, default='SENTINEL', help='Sensor')
    parser.add_argument('--AMP', type=bool, default=True, help='Mixed Precision Training')
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()
    pipeline = ObjectDetectionPipeline(batch_size=args.batch_size, 
                                       band=args.band, 
                                       seed=args.seed, 
                                       learning_rate=args.learning_rate, 
                                       max_epochs=args.max_epochs, 
                                       amp=args.AMP,
                                       sensor='SENTINEL')
    

    
    pipeline.build() # Builds the pipeline

    pipeline.train()
    pipeline.test()