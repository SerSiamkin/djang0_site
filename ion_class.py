#!/usr/bin/env python
# coding: utf-8

# Alexey V. Oinats (2024)
# oinats@iszf.irk.ru


import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator, LogFormatter)

from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from datetime import datetime as dt
import os.path
import copy
from typing import BinaryIO

class Ionogram():
    """
    Класс ионограмм
    readion - чтение файлов ионограмм в формате "dat"
    readion_binary - чтение ионограмм в формате BinaryIO
    writeion - запись ионограммы в файл в формате "dat"
    get_param - выдача значения параметра из паспорта ионограммы
    get_passport - выдача паспорта ионограммы
    print_passport - печать паспорта ионограммы
    get_frequences - выдача массива частот
    get_heights - выдача массива виртуальных высот
    get_dimension - выдать разрешение по шкале высот
    get_ionogram - выдача матрицы ионограммы
    plot_ionogram - построение ионограммы
    plot_ionogram_rect - построение ионограммы патчами
    plot_noise - построение частотной зависимости "шума"

    Вторичная обработка
    do_medfilt2d - фильтрация медианным 2D фильтром для удаления шумов
    do_cellular_automaton - клеточный автомат для выделения точек со значимой амплитудой
    
    Экспериментальные функции (пользоваться осторожно!)
    __eq__ и __ne__- "сравнение" ионограмм: операции "==" и "!="
    __add__ и __iadd__- "сложение" ионограмм: операции "+" и "+="
    __truediv__ - деление ионограммы на скалярное число "/scalar"
    """
    light_velocity = 3.0e+8
    nheights = 512
    nfrequences = 512   
    __delimeter = '\x00\x00\x00\x00'
    __cluster_flag = b'\x80'
    __align = 1

    
    def __init__(self):
        self.noise = []
        self.echoes = []
        self.ionogram_matrix = []
        self.medfilt2d = []
        self.cellular_automaton = []
        self.first_delay = 0
        self.nheights = 512
        self.maxheight = 0
        self.imaxheight = 0
        self.nfrequences = 512
        self.__parameters = {
            'date':       {'value': '', 'type': 's', 'description': 'Дата',                                 'units': ''},
            'time':       {'value': '', 'type': 's', 'description': 'Время начала сеанса',                  'units': ''},
            'path':       {'value': '', 'type': 's', 'description': 'Трасса зондирования',                  'units': ''}, 
            'mode':       {'value': '', 'type': 's', 'description': 'Режим',                                'units': ''}, 
            'delay':      {'value': 0, 'type': 'd', 'description': 'Задержка',                             'units': ''}, 
            'freq0':      {'value': 0, 'type': 'd', 'description': 'Начальная частота',                    'units': ''}, 
            'freqN':      {'value': 0, 'type': 'd', 'description': 'Конечная частота',                     'units': ''}, 
            'chirp_rate': {'value': 0, 'type': 'd', 'description': 'Скорость сканирования',                'units': ''}, 
            'band_width': {'value': 0, 'type': 'd', 'description': 'Полоса анализа',                       'units': ''}, 
            'antenna':    {'value': '', 'type': 's', 'description': 'Антенна',                              'units': ''}, 
            'adc':        {'value': 0, 'type': 'd', 'description': 'Номер АЦП',                            'units': ''}, 
            'freq_step':  {'value': 0, 'type': 'd', 'description': 'Дискретность по частоте зондирования', 'units': ''}, 
            'ampl_coef':  {'value': 0, 'type': 'd', 'description': 'Коэффициент усиления',                 'units': ''},
            'latitude':   {'value': 0, 'type': 'f', 'description': 'Широта пункта приёма',                 'units': ''}, 
            'longitude':  {'value': 0, 'type': 'f', 'description': 'Долгота пункта приёма',                'units': ''}, 
            'height':     {'value': 0, 'type': 'f', 'description': 'Высота пункта приёма',                 'units': ''},
        }
        self.__ionogram_loaded = False
        
    def __str__(self):
        info = ''
        if(self.__ionogram_loaded):
            info = "Current:\n {:s}\n{:s} UT".format(self.__parameters['path']['value'], dt.strftime(self.date_time, '%d.%m.%Y %H:%M:%S'))
        return "This class provides read and plot ionogram in DAT-format\n{:s}".format(info)

    def __eq__(self, other):
        res = bool(1)
        if(len(self.noise) != len(other.noise)):
            res = bool(0)
        if(self.__parameters['freq0']['value'] != other.__parameters['freq0']['value']):
            res = bool(0)
        if(self.__parameters['freqN']['value'] != other.__parameters['freqN']['value']):
            res = bool(0)
        if(self.__parameters['freq_step']['value'] != other.__parameters['freq_step']['value']):
            res = bool(0)
        if(np.abs((self.maxheight / self.imaxheight) - (other.maxheight / other.imaxheight)) > 0.01):
            res = bool(0)
        if(self.delay != other.delay):
            res = bool(0)
        return res

    def __ne__(self, other):
        return not self.__eq__(other)

    def __truediv__(self, scalar):
        c = copy.deepcopy(self)
        index_max = np.argmax(np.array(c.echoes)[:, 1])
        for i,e in enumerate(c.noise):
            c.noise[i][1] -= 20*np.log10( scalar )
            if(c.noise[i][1] < 0):
                # print('WARNING: The noise become negative!')
                c.noise[i][1] = 0
        for i,e in enumerate(c.echoes):
            # c.echoes[i][2] -= 20*np.log10( scalar )
            if(i != index_max):
                c.echoes[i][2] -= 20*np.log10( scalar )
            else:
                c.echoes[i][2] = 1
            if(c.echoes[i][2] < 0):
                # print('WARNING: The amplitude become negative!')
                c.echoes[i][2] = np.nan
        return c

    def __iadd__(self, other):
        for i,e in enumerate(other.noise):
            self.noise[i][1] = 20*np.log10( 10**(self.noise[i][1]/20) + 10**(e[1]/20) )
            # print(i,e)
        for i,e in enumerate(other.echoes):    
            echoe_exist = bool(0)            
            for i1,e1 in enumerate(self.echoes): 
                if(e1[:2] == e[:2]):
                    # print(e, e1)
                    self_amp = 10**(e1[2]/20)
                    other_amp = 10**(e[2]/20)
                    # self.echoes[i1][2] = 20*np.log10( self_amp + other_amp )
                    
                    sum_value = 20*np.log10( self_amp + other_amp )
                    # print(e1[1])
                    if(sum_value > self.noise[e1[0]][1]):
                        self.echoes[i1][2] = sum_value
                    else:
                        self.echoes.pop(i1)
                        
                    echoe_exist = bool(1)
                    break
            if(not echoe_exist):
                # self.echoes.append([e[0], e[1], e[2]])

                if(e[2] > self.noise[e[0]][1]):
                        self.echoes.append([e[0], e[1], e[2]])
        # print(len(self.echoes))
        return self

    def __add__(self, other):
        c = copy.deepcopy(self)
        for i,e in enumerate(other.noise):
            c.noise[i][1] = 20*np.log10( 10**(c.noise[i][1]/20) + 10**(e[1]/20) )
        for i,e in enumerate(other.echoes):    
            echoe_exist = bool(0)            
            for i1,e1 in enumerate(c.echoes): 
                if(e1[:2] == e[:2]):
                    a_amp = 10**(e1[2]/20)
                    other_amp = 10**(e[2]/20)
                    # c.echoes[i1][2] = 20*np.log10( a_amp + other_amp )
                    
                    sum_value = 20*np.log10( a_amp + other_amp )
                    if(sum_value > c.noise[e1[0]][1]):
                        c.echoes[i1][2] = sum_value
                    else:
                        c.echoes.pop(i1)
                        
                    echoe_exist = bool(1)
                    break
            if(not echoe_exist):
                # c.echoes.append([e[0], e[1], e[2]])

                if(e[2] > c.noise[e[0]][1]):
                        c.echoes.append([e[0], e[1], e[2]])
        return c

    def __get_param(self, parname):
        return self.__parameters[parname]['value']

    def get_passport(self):
        p = ''
        for par in self.__parameters:
            p += '{:s}:'.format(self.__parameters[par]['description'])
            if('s' in self.__parameters[par]['type']):
                p += ' {:s}'.format(self.__parameters[par]['value'])
            if('f' in self.__parameters[par]['type']):
                p += ' {:.4f}'.format(self.__parameters[par]['value'])
            if('d' in self.__parameters[par]['type']):
                p += ' {:d}'.format(self.__parameters[par]['value'])
            if(self.__parameters[par]['units'] != ''):
                p += ' {:s}'.format(self.__parameters[par]['units'])
            p += "\n"
        p += "\n"
        return p

    def print_passport(self):
        p = Ionogram.get_passport(self)
        return print(p)

    def readion(self, filename: str = None):
        """
        Функция чтения файла ионограммы
        filename - путь к "dat"-файлу
        """
        if(filename is not None):
            f = open(filename, mode="rb")
            self.readion_binary(f)
            f.close()
        else:
            print('You should indicate ionogram filename to read!')

    def readion_binary(self, binarydata: BinaryIO):
        """
        Функция чтения ионограммы в бинарном виде
        binarydata - объект BinaryIO
        """
        self.__data = binarydata.read()
        b = self.__data.decode('cp866')
        self.__passport = b[:b.find(self.__delimeter)]
        self.__ionogram = self.__data[b.find(self.__delimeter) + 4:]
        Ionogram.__parse_passport(self)
        Ionogram.__parse_ionogram(self)
        # self.nheights = Ionogram.get_dimension(self)
        self.imaxheight = np.max(np.array(self.echoes)[:, 1])
        self.nheights = self.imaxheight+1
        self.dheight = self.maxheight / self.imaxheight / 1000
        self.__ionogram_loaded = True
        if(np.array(self.noise).size == 0):
            print('WARNING: Ionogram does not contain noise data')
            
    def writeion(self, filename: str = None, rewrite=False):
        """
        Функция записи файла ионограммы
        filename - новый путь к "dat"-файлу
        rewrite - if file exists this flag allows to rewrite it
        """
        if(self.__ionogram_loaded is not True):
            print('Ionogram is empty!')
            return
        if(filename is not None):            
            if(os.path.isfile(filename) is True and rewrite is False):
                print('Destination file already exists!')
                return
            with open(filename, "wb") as f:
                p = Ionogram.get_passport(self)
                b = p.encode('cp866')
                f.write(b)
                f.write(bytearray(self.__delimeter, 'cp866'))
                
                amp = Ionogram.get_ionogram(self)
                noise = np.array(self.noise)
                # print(noise)
                # return
                for ifn in range(self.nfrequences):
                    f.write((ifn+1 | 32768).to_bytes(2, 'big'))
                    f.write(Ionogram.__align.to_bytes(2, 'big'))
                    # print(noise[ifn, 1])
                    f.write(int(noise[ifn, 1].item()).to_bytes(2, 'big'))
                    f.write(Ionogram.__align.to_bytes(2, 'big'))
                    for it in range(self.nheights):
                        if(~np.isnan(amp[it, ifn])):
                            f.write(int(amp[it, ifn].item()).to_bytes(2, 'big'))
                            f.write(it.to_bytes(2, 'big'))
                f.close()
        else:
            print('You should indicate ionogram filename to write!')

    def __parse_passport(self):
        for v in self.__passport.splitlines():
            for par in self.__parameters:
                if(v.find(self.__parameters[par]['description']) != -1):
                    idx1 = v.index(": ") + 2
                    if('s' in self.__parameters[par]['type']):
                        self.__parameters[par]['value'] = v[idx1:].strip()
                    else:
                        while(v[idx1:].find(" ") == 0):
                            idx1 += 1
                        if(v[idx1:].find(" ") == -1):
                            self.__parameters[par]['value'] = v[idx1:].strip()
                        else:
                            idx2 = v[idx1:].index(" ")
                            self.__parameters[par]['value'] = v[idx1: idx1 + 1 + idx2].strip()
                            self.__parameters[par]['units'] = v[idx1 + idx2 + 1:].strip()
        
        Ionogram.__parse_date_time(self)
        
        for par in self.__parameters:
            if('f' in self.__parameters[par]['type']):
                self.__parameters[par]['value'] = float(self.__parameters[par]['value'])
            if('d' in self.__parameters[par]['type']):
                self.__parameters[par]['value'] = int(self.__parameters[par]['value'])
        
        self.nfrequences = int((self.__parameters['freqN']['value'] - self.__parameters['freq0']['value'])                                / self.__parameters['freq_step']['value'])
        self.first_delay = self.__parameters['delay']['value'] * Ionogram.light_velocity / 1000 / 1000
        if(self.__parameters['mode']['value'] == "НЗ"):
            self.maxheight = Ionogram.light_velocity * self.__parameters['band_width']['value']                                 / self.__parameters['chirp_rate']['value'] / 1000
        else:
            self.maxheight = Ionogram.light_velocity * self.__parameters['band_width']['value']                                 / self.__parameters['chirp_rate']['value'] / 2.0 / 1000
        
    def __parse_date_time(self):
        dt_string = '{:s} {:s}'.format(self.__parameters['date']['value'], self.__parameters['time']['value'][0:8])
        self.date_time = dt.strptime(dt_string, '%d.%m.%Y %H:%M:%S')

    def __check_new_cluster(value, flag):
        result = (int.from_bytes(value[0:1],'big') & int.from_bytes(flag, 'big')).to_bytes(max(len(value[0:1]), len(flag)), 'big')
        if(result == flag):
            return True
        return False
            
    def __parse_ionogram(self):
        ifn_ = -1
        for ib in np.arange(0,len(self.__ionogram),4):
            if(Ionogram.__check_new_cluster(self.__ionogram[ib:ib+2], Ionogram.__cluster_flag)):
                ifn = int.from_bytes(self.__ionogram[ib:ib+2], 'big') & 32767          
                ifn_ += 1
                if(int.from_bytes(self.__ionogram[ib+2:ib+4], 'big') != 1):
                    print('ERROR in align!')
            else:
                amp = int.from_bytes(self.__ionogram[ib:ib+2], 'big')
                it  = int.from_bytes(self.__ionogram[ib+2:ib+4], 'big')
                # print(ifn_, it, amp)
                if(it == 1):
                    self.noise.append([ifn_, amp])
                else:
                    self.echoes.append([ifn_, it, amp])

    def get_frequences(self):
        return np.array([(self.__parameters['freq0']['value'] + i * self.__parameters['freq_step']['value'])                          / 1000 for i in range(self.nfrequences + 1)])
    
    def get_heights(self):
        return np.array([self.first_delay + i * self.dheight for i in range(self.nheights + 1)])
        # return np.array([self.first_delay + (i - 1) * self.dheight for i in range(self.nheights + 1)]) #This is for corespondence with Grozov&Ponomarchuk programs

    def get_dimension(self):
        if(self.imaxheight > 512):
            return 1024
        return 512
        
    # def get_ionogram(self, value=None, recalc=False):
    #     if(not np.any(self.ionogram_matrix) or recalc):
    #         print('Recalculating self.ionogram_matrix')
    #         if(value is None):
    #             value = np.nan
    #         self.ionogram_matrix = np.full((self.nheights, self.nfrequences,), fill_value=value, dtype=np.float64)
    #         for x in self.echoes:
    #             if(x[1] < self.nheights):
    #                 self.ionogram_matrix[x[1], x[0]] = x[2]
    #     return self.ionogram_matrix
    
    def get_ionogram(self, recalc=False):
        if(not np.any(self.ionogram_matrix) or recalc):
            # print('Recalculating self.ionogram_matrix')
            self.ionogram_matrix = np.full((self.nheights, self.nfrequences,), fill_value=np.nan, dtype=np.float64)
            for x in self.echoes:
                if(x[1] < self.nheights):
                    self.ionogram_matrix[x[1], x[0]] = x[2]
        return self.ionogram_matrix
    
    def do_medfilt2d(self, size=3, order=1, recalc=False):
        """
        Медианный 2D фильтр для удаления шумов
        """
        if(not np.any(self.medfilt2d) or recalc):
            # print('Recalculating self.medfilt2d')
            echoes_to_filt = np.copy(self.get_ionogram(recalc=recalc))
            echoes_to_filt[np.isnan(echoes_to_filt)] = 0
            for i in range(order):
                self.medfilt2d = sc.signal.medfilt2d(echoes_to_filt, kernel_size=size)
                echoes_to_filt = np.copy(self.medfilt2d)
        return self.medfilt2d

    def do_cellular_automaton(self, freq_size=3, height_size=3, recalc=False):
        """
        Клеточный автомат для выделения точек со значимой амплитудой
        """
        if(not np.any(self.cellular_automaton) or recalc):
            # print('Recalculating self.cellular_automaton')
            echoes_tmp = self.do_medfilt2d(size=3, order=1, recalc=recalc)
            cellular = []
            norm = 1.0 / freq_size / height_size
            aver_amp_array = sc.ndimage.uniform_filter(echoes_tmp, size=[height_size, freq_size], mode='constant')
            for i,f in enumerate(self.get_frequences()[:-1]):
                row_3s = 3.0*np.std(echoes_tmp[:,max(i-1,0):min(i-1+freq_size,echoes_tmp.shape[1])])
                cmax_amp = 0
                incr_flag = True                
                for j,h in enumerate(self.get_heights()[:-1]):
                    if(aver_amp_array[j, i] > row_3s):
                        if(cmax_amp <= aver_amp_array[j, i]):
                            incr_flag = True
                            cmax_amp = aver_amp_array[j, i]
                        else:
                            if(incr_flag and echoes_tmp[j, i] != 0):
                                cellular.append([i, j, int(echoes_tmp[j, i])])
                                incr_flag = False
                    else:
                        cmax_amp = 0
            self.cellular_automaton = np.full((self.nheights, self.nfrequences,), fill_value=np.nan, dtype=np.float64)
            for x in cellular:
                if(x[1] < self.nheights):
                    self.cellular_automaton[x[1], x[0]] = x[2]      
        return self.cellular_automaton
                
    def plot_ionogram(self, mode=None, ax=None, freq_min=None, freq_max=None, height_min=None, height_max=None, title=True, fontsize=16):
        """
        Функция построения графика ионограммы
        mode - режим построение после медианного фильтра и выделения точек со значимой амплитудой
        ax - идентификатор осей координат
        freq_min - левая граница на оси частот
        freq_max - правая граница на оси частот
        height_min - нижняя граница на оси высот/дальности
        height_max - верхняя граница на оси высот/дальности
        title - флаг вывода подзаголовка (True/False - c/без заголовка)
        """
        title_mode = ''

        if(mode is not None):
            if('medfilt2d' in mode):
                z = self.do_medfilt2d()
                title_mode = ' [мед. фильтр]'
                
            if('cellular_automaton' in mode):
                z = self.do_cellular_automaton()
                title_mode = ' [знач. точки]'
        else:
            z = self.get_ionogram()
        
        x = self.get_frequences()
        y = self.get_heights()
        
        if not ax:
            ax = plt.gca()

        axins = inset_axes(
            ax,
            width="3%",  # width: 5% of parent_bbox width
            height="100%",  # height: 50%
            loc="lower left",
            bbox_to_anchor=(1.02, 0., 1, 1),
            bbox_transform=ax.transAxes,
            borderpad=0,
        )
        
        ax.set_ylabel( 'range, km', fontsize=fontsize )
        if(self.__parameters['mode'] == 'ВЗ'):
            ax.set_ylabel( 'virtual height, km', fontsize=fontsize )
        y1 = self.first_delay
        y2 = np.nanmax(y)
        if(height_min is not None):
            y1 = height_min
        if(height_max is not None):
            y2 = height_max
            
        ax.set_ylim([y1, y2])
        ax.set_yticks(np.arange(y1, y2, 100))
        
        ax.set_xlabel( 'frequency, MHz', fontsize=fontsize )
        
        x1 = np.nanmin(x)
        x2 = np.nanmax(x)
        if(freq_min is not None):
            x1 = freq_min
        if(freq_max is not None):
            x2 = freq_max
            
        ax.set_xlim([x1, x2])
        ax.set_xticks(np.arange(x1, x2, 1))
        
        if(title):
            pictitle = '{:s}{:s}\n{:s} UT'.format(self.__parameters['path']['value'], title_mode, dt.strftime(self.date_time, '%d.%m.%Y %H:%M:%S'))
            # plt.suptitle(pictitle, y=0.94, fontsize=16, fontdict={'weight': 'normal'})
            ax.set_title(pictitle, fontsize=fontsize, fontdict={'weight': 'normal'})
            
        ax.tick_params(axis='both', which='both', labelsize=(fontsize - 2), direction='in', length=5)
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        
        cp = ax.pcolormesh(x, y, z, cmap='jet', vmin = np.nanmin(z), vmax = np.nanmax(z))

        cbar = plt.colorbar(cp, ax=ax, cax=axins, ticks = np.arange(np.nanmin(z), np.nanmax(z), 10))
        cbar.ax.set_title('Amplitude, dB', fontdict = {'fontsize': fontsize, 'fontweight': 'normal'}, pad = 10)
        cbar.ax.tick_params(labelsize = (fontsize - 2))

        ax.grid(which='major', axis='both', linestyle = ':', color="black")

    def plot_noise(self, ax=None, freq_min=None, freq_max=None, title=False, fontsize=16):
        """
        Функция построения графика шума в зависимости от частоты
        ax - идентификатор осей координат
        freq_max - правая граница на оси частот
        title - флаг вывода подзаголовка (True/False - c/без заголовка)
        """
        y = np.array(self.noise)
        x = self.get_frequences()

        if(y.size == 0):
            print('Ionogram does not contain noise data')
            return
        
        if not ax:
            ax = plt.gca()

        ax.set_ylabel( 'noise, dB?', fontsize=fontsize )
        ax.set_xlabel( 'frequency, MHz', fontsize=fontsize )
        
        x1 = np.nanmin(x)
        x2 = np.nanmax(x)
        if(freq_min is not None):
            x1 = freq_min
        if(freq_max is not None):
            x2 = freq_max
            
        ax.set_xlim([x1, x2])
        ax.set_xticks(np.arange(x1, x2, 1))
            
        if(title):
            pictitle = '{:s} {:s} UT'.format(self.__parameters['path']['value'], dt.strftime(self.date_time, '%d.%m.%Y %H:%M:%S'))
            plt.suptitle(pictitle, y=1, fontsize=fontsize, fontdict={'weight': 'normal'})
        ax.tick_params(axis='both', which='both', labelsize=(fontsize - 2), direction='in', length=5)
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))

        ax.plot(x[:-1], y[:,1], 'bo', label='', markersize=2)
        ax.grid(which='major', axis='both', linestyle = ':', color="black")

    def plot_ionogram_rect(self, ax=None, freq_min=None, freq_max=None, height_min=None, height_max=None, title=True, fontsize=16):
        """
        Функция построения графика ионограммы (патчами - прямоугольниками)
        ax - идентификатор осей координат
        freq_min - левая граница на оси частот
        freq_max - правая граница на оси частот
        height_min - нижняя граница на оси высот/дальности
        height_max - верхняя граница на оси высот/дальности
        title - флаг вывода подзаголовка (True/False - c/без заголовка)
        """
        z = self.get_raw_ionogram()
        x = self.get_frequences()
        y = self.get_heights()
        
        if not ax:
            ax = plt.gca()

        axins = inset_axes(
            ax,
            width="3%",  # width: 5% of parent_bbox width
            height="100%",  # height: 50%
            loc="lower left",
            bbox_to_anchor=(1.02, 0., 1, 1),
            bbox_transform=ax.transAxes,
            borderpad=0,
        )

        ax.set_ylabel( 'range, km', fontsize=fontsize )
        if(self.__parameters['mode'] == 'ВЗ'):
            ax.set_ylabel( 'virtual height, km', fontsize=fontsize )
        y1 = self.first_delay
        y2 = np.nanmax(y)
        if(height_min is not None):
            y1 = height_min
        if(height_max is not None):
            y2 = height_max
            
        ax.set_ylim([y1, y2])
        ax.set_yticks(np.arange(y1, y2, 100))
        
        ax.set_xlabel( 'frequency, MHz', fontsize=fontsize )
        
        x1 = np.nanmin(x)
        x2 = np.nanmax(x)
        if(freq_min is not None):
            x1 = freq_min
        if(freq_max is not None):
            x2 = freq_max
            
        ax.set_xlim([x1, x2])
        ax.set_xticks(np.arange(x1, x2, 1))
        
        if(title):
            pictitle = '{:s}\n{:s} UT'.format(self.__parameters['path']['value'], dt.strftime(self.date_time, '%d.%m.%Y %H:%M:%S'))
            plt.suptitle(pictitle, y=0.94, fontsize=fontsize, fontdict={'weight': 'normal'})
            
        ax.tick_params(axis='both', which='both', labelsize=(fontsize - 2), direction='in', length=5)
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))

        cmap = ListedColormap(plt.get_cmap('jet')(np.linspace(0, 1, 256)))  # skip too light colors
        amplitudes = np.array([e[2] for e in z])
        norm = plt.Normalize(amplitudes.min(), amplitudes.max())
        echoes_rect = [Rectangle((x[e[0]], y[e[1]]), self.__parameters['freq_step']['value'] / 1000, self.maxheight / self.imaxheight / 1000, color=cmap(norm(e[2]))) for e in z]
        echoes_color = [cmap(norm(e[2])) for e in z]

        pc = PatchCollection(echoes_rect, facecolor=echoes_color, alpha=1.0)
        ax.add_collection(pc)

        cbar = plt.colorbar(ScalarMappable(cmap=cmap, norm=norm), ax=ax, cax=axins, ticks = np.arange(np.nanmin(amplitudes), np.nanmax(amplitudes), 10))
        cbar.ax.set_title('Amplitude, dB', fontdict = {'fontsize': fontsize, 'fontweight': 'normal'}, pad = 10)
        cbar.ax.tick_params(labelsize = (fontsize - 2))

        ax.grid(which='major', axis='both', linestyle = ':', color="black")






