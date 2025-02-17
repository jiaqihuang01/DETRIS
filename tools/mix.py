import lmdb
import shutil
import os
import pyarrow as pa
from tqdm import tqdm

def loads_pyarrow(buf):
    """
    Args:
        buf: the output of `dumps`.
    """
    return pa.deserialize(buf)


def raw_reader(path):
    with open(path, 'rb') as f:
        bin_data = f.read()
    return bin_data


def dumps_pyarrow(obj):
    """
    Serialize an object.
    Returns:
        Implementation-dependent bytes-like object
    """
    return pa.serialize(obj).to_buffer()

def mix_lmdb_databases(output_path, *input_paths):
    db = lmdb.open(output_path,
                   map_size=1099511627776 * 2, readonly=False, subdir=False,
                   meminit=False, map_async=True)
    txn_out = db.begin(write=True)
    idx = 0
    for input_path in input_paths:
        dataset = input_path.split("/")[-2]
        print(f"Merging {input_path}")
        env_in = lmdb.open(input_path, subdir=False, readonly=True)
        txn_in = env_in.begin()
        cursor = txn_in.cursor()
        total_records = txn_in.stat()['entries']
        for key, value in tqdm(cursor, total=total_records, desc=input_path):
            if key.startswith(b"__"):
                continue
            value = loads_pyarrow(value)
            value['seg_id'] = f"{dataset}_{value['seg_id']}"
            txn_out.put(u'{}'.format(idx).encode('ascii'), dumps_pyarrow(value))
            idx += 1
        txn_in.abort()
        env_in.close()
        
    txn_out.commit()
    keys = [u'{}'.format(k).encode('ascii') for k in range(idx + 1)]
    with db.begin(write=True) as txn:
        txn.put(b'__keys__', dumps_pyarrow(keys))
        txn.put(b'__len__', dumps_pyarrow(len(keys)))

    print("Flushing database ...")
    db.sync()    
    db.close()

output_dir = './datasets/lmdb/refcoco_mixed/'
splits = ['train', 'val']
for split in splits:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = f"{output_dir}/{split}.lmdb"
    input_dirs = ['./datasets/lmdb/refcoco', './datasets/lmdb/refcoco+', './datasets/lmdb/refcocog_u']
    input_paths = [f"{input_dir}/{split}.lmdb" for input_dir in input_dirs]
    mix_lmdb_databases(output_path, *input_paths)
