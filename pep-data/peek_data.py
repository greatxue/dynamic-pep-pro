import lmdb
import pickle

env = lmdb.open("/mlx_devbox/users/xuezhongkai/playground/dynamic-pep-pro/pep-data/PepMerge/pep_pocket_train_structure_cache.lmdb", 
                readonly=True, lock=False, subdir=False)

with env.begin() as txn:
    cursor = txn.cursor()
    
    for key, value in cursor:
        print("KEY:", key)
        
        try:
            data = pickle.loads(value)
            print(type(data))
            print(data.keys() if isinstance(data, dict) else data)
        except:
            print(value[:200])  # 看前200字节
        
        break