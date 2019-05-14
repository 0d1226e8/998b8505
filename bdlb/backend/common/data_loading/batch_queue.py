from __future__ import print_function

try:
  import Queue as queue
except ImportError:
  import queue
import threading


def batch_queue(dataset, indices, batch_size, max_q_size=20, nb_worker=8):
  """Creates queue and threads used internally
  by a dataset object to concurrently load
  batches from disk.

  Args:
    dataset : Dataset
      Dataset object to load batches from.

    indices : numpy array of ints
      Indices of batches to load.

    batch_size : int
      Size of batches.

    max_q_size : int, 20 by default
      Max number of batches concurrently in the queue.

    nb_worker : int, 8 by default
      Number of workers.
  """
  # Initialize member variables.
  generator_threads = []

  q_indices = queue.Queue()
  q_out = queue.Queue(maxsize=max_q_size)
  stop = threading.Event()

  # Initialize queue of batches of indices, used by threads/processes to determine
  # the indices of images to load for their next batch.
  for i in range(0, len(indices), batch_size):
    batch = indices[i:i + batch_size]
    q_indices.put(batch)

  if len(indices) % batch_size != 0:
    final_batch = indices[-(len(indices) % batch_size):]
    q_indices.put(final_batch)

  def data_generator_task():
    """Task for each thread/process."""
    while not stop.is_set() and not q_indices.empty():
      try:
        idxs = q_indices.get(block=False)
        image_batch = dataset._get_images(idxs)
        if dataset.y is not None:
          image_batch = (image_batch, dataset._get_labels(idxs))
        while True and not stop.is_set():
          try:
            q_out.put(image_batch, block=True, timeout=0.1)
            break
          except queue.Full:
            continue
      except queue.Empty:
        stop.set()
        break
      except Exception as ex:
        stop.set()
        raise ex

  try:
    for i in range(nb_worker):
      thread = threading.Thread(target=data_generator_task)
      generator_threads.append(thread)
      thread.daemon = True
      thread.start()
  except Exception as ex:
    stop.set()
    raise ex

  return q_out, stop
