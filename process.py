#!/usr/bin/env python
# coding: utf-8

# Nikita A. Gromik (2024)
# gromik@iszf.irk.ru

# Этот файл содержит код, для выполнения логики накопления 
# пакета ионограмм в многопоточном режиме.

import re
import os
import copy
import argparse
import threading
from typing import Optional
from datetime import datetime
from dotenv import load_dotenv
from multiprocessing import Pool, freeze_support, cpu_count, Value

from ion_class import Ionogram

def get_prefix():
  return "[{:s}\t{:06d}] >>".format(datetime.now().strftime('%d.%m.%Y %H:%M:%S'), threading.get_ident())

def process(year, month, day, h, ms, me):
  global s_start
  global folder_in
  global folder_out

  dpath = folder_out(year, month, day)
  ofname = os.path.join(dpath, "{:02d}_{:02d}_{:02d}_{:02d}_{:02d}.dat".format(month, day, h, ms, s_start))
  if os.path.exists(ofname):
    with progress_value.get_lock():
      progress_value.value += 1
      # print(get_prefix(), "File \"{:s}\" exists. Skip! Progress: {:.2f}% ({:d}/{:d})".format("{:02d}_{:02d}_{:02d}_{:02d}_{:02d}.dat".format(month, day, h, ms, s_start), float(progress_value.value / progress_total.value) * 100.0, progress_value.value, progress_total.value))
    return

  nion = 0
  ions = []
  print(get_prefix(), "Start process from {:02d}.{:02d} {:02d}:{:02d} - {:02d}:{:02d}. Progress: {:.2f}% ({:d}/{:d})".format(day, month, h, ms, h, min(59, me), float(progress_value.value / progress_total.value) * 100.0, progress_value.value, progress_total.value))
  for m in range(ms, me):
    for s in range(s_start, 60, 15):
      fname = os.path.join(folder_in(year, month, day), "{:02d}_{:02d}_{:02d}_{:02d}_{:02d}.dat".format(month, day, h, m, s))
      if not os.path.exists(fname):
        print(get_prefix(), "File {:s} doesn't exists!".format(fname))
        continue
      # print('Reading file: ', fname)        
      
      ions.append(Ionogram())
      ions[nion].readion(fname)
      nion += 1

  with progress_value.get_lock():
    progress_value.value += 1

  if len(ions) <= 0:
    print(get_prefix(), "Empty ionograms... Skip!")
    return
  else:
    print(get_prefix(), "Saving... Please, wait...")

  # Суммируем ионограммы
  sum_ion = copy.deepcopy(ions[0])
  for ni,ii in enumerate(ions):
    if(ni > 0):
        sum_ion += ii

  # Делим на количество, чтобы найти усредненную амплитуду
  aion = sum_ion / len(ions)
  if not os.path.exists(dpath):
    os.makedirs(dpath)
  aion.writeion(ofname)
  print(get_prefix(), "Saved to: {:s}. Progress: {:.2f}% ({:d}/{:d})".format(ofname, float(progress_value.value / progress_total.value) * 100.0, progress_value.value, progress_total.value))

def init_globals(pv, pt):
  global progress_value
  global progress_total

  progress_value = pv
  progress_total = pt

  global folder_in
  folder_in = lambda y, m, d: str(os.environ.get('FOLDER_IN')).format(
    YEAR = str(y).zfill(2),
    MONTH =  str(m).zfill(2),
    DAY = str(d).zfill(2),
  )

  global folder_out
  folder_out = lambda y, m, d: str(os.environ.get('FOLDER_OUT')).format(
    YEAR = str(y).zfill(2),
    MONTH =  str(m).zfill(2),
    DAY = str(d).zfill(2),
  )

  global dm
  dm = int(os.environ.get('DELTA_MINUTES'))

  global date_from
  date_from = list(map(lambda x: int(x), str(os.environ.get('DATE_FROM')).split('-')))

  global date_to
  date_to = list(map(lambda x: int(x), str(os.environ.get('DATE_TO')).split('-')))

  global s_start
  s_start = get_s_start()

def check_resolution() -> list[int, list[str]]:
  """
  Проверяет разрешение ионограмм, если они разносортные - выдает предупреждение
  """

  global s_start
  global date_to
  global date_from
  global folder_in

  result = [None, []] # Разрешение первого, Список файлов с иным разрешением
  for year in range(date_from[0], date_to[0] + 1):
    for month in range(date_from[1], date_to[1] + 1):
      for day in range(date_from[2], date_to[2] + 1):
        for h in range(0, 24):
          for m in range(0, 60, dm):
            for s in range(s_start, 60, 15):
              ifname = "{:02d}_{:02d}_{:02d}_{:02d}_{:02d}.dat".format(month, day, h, m, s)
              ifpath = os.path.join(folder_in(year, month, day), ifname)
              if not os.path.exists(ifpath):
                continue

              ionogram = Ionogram()
              ionogram.readion(ifpath)
              res = ionogram.get_dimension()
              if not result[0]:
                result[0] = res
              elif res != result[0]:
                result[1].append(ifname)
  return result

def check_envs() -> bool:
  print('/*/* ENVIRONMENT */*/')

  check_dict = {
    "DATE_FROM": "Дата начала обработки",
    "DATE_TO": "Дата завершения обработки",
    "DELTA_MINUTES": "Накопление в минутах",
    "FOLDER_IN": "Путь исходных данных",
    "FOLDER_OUT": "Путь выходных данных",
  }
  for k, v in check_dict.items():
    val = os.environ.get(k)
    if not val:
      print(f'*** Не найдено значение {v} ({k}). Возможно вы не заполнили файл .env. Пример заполнения находится в .env.sample ***')
      return False
    print(v, val)
  print('/*/* ENVIRONMENT */*/')
  return True

def get_s_start() -> int:
  """
  Определяет начальную секунду,
  Поскольку не все ионограммы считаются с 00:00, некоторые могут начинаться с 00:03
  """
  fname = os.listdir(folder_in(date_from[0], date_from[1], date_from[2]))[0]

  m = re.match("^\d{2}_\d{2}_\d{2}_\d{2}_(\d{2})\.dat$", fname)
  if m:
    return int(m.group(1))

  return 0


def main(check: Optional[bool] = None):
  load_dotenv()

  if not check_envs():
    return

  progress_value = Value('i', 0)
  progress_total = Value('i', 1)

  init_globals(0, 0)
  args = []
  for year in range(date_from[0], date_to[0] + 1):
    for month in range(date_from[1], date_to[1] + 1):
      for day in range(date_from[2], date_to[2] + 1):
        for h in range(0, 24):
          for m in range(0, 60, dm):
              args.append((year, month, day, h, m, m + dm))

  print("Args: [", args[0], args[1], "...", args[-2], args[-1], "]. Total: ", len(args))

  if check:
    res_warn = check_resolution()
    print(f'Первая обрабатываемая ионограмма с разрешением {res_warn[0]} точек')
    if res_warn[1]:
      print('*** Найдены ионограммы, разрешение которых отличается от первой! ***')
      for fname in res_warn[1]:
        print(f'*** {fname} ***')
  else:
    print('Ионограммы не будут проверяться по разрешению высот. Для включения настройки запустите обработку с флагом -c')

  progress_total.value = len(args)

  print("Запуск программы в", cpu_count(), "потоков")
  with Pool(initializer=init_globals, initargs=(progress_value, progress_total,)) as pool:
    pool.starmap(process, args)
    print(get_prefix(), "Завершено", progress_value.value, ' / ', progress_total.value)


if __name__=="__main__":
  parser = argparse.ArgumentParser()

  parser.add_argument('-ci', '--check', required=False, type=bool)
  args = parser.parse_args()

  print(get_prefix(), "Запуск...")
  freeze_support()
  main(check=args.check)
