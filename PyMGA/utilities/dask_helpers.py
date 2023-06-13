from dask.distributed import Client, LocalCluster
from dask_jobqueue import SLURMCluster


def start_dask_cluster(workers=32, try_slurm=True):
    try: 
        client = Client()
        cluster = None
    except Exception as e: 
        print('Error', e)
        if try_slurm:
            try: 
                cluster = SLURMCluster(log_directory='logs/',
                                       walltime='24:00:00')
                print('started slurm cluster')
                cluster.scale(workers)
            except Exception as e:
                print('Error', e)
                cluster = LocalCluster(scheduler_port=8786,
                                       processes=True)
                print('started local cluster')
        else:
            cluster = LocalCluster(scheduler_port=8786,
                                   processes=True)
            print('started local cluster')
        client = Client(cluster)
    return cluster, client

