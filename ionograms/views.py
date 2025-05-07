import os
from django.shortcuts import render
import plotly.graph_objects as go
import numpy as np
from ion_class import Ionogram
from datetime import datetime
from django.http import JsonResponse
import glob

def list_files(request):
    base_path = request.GET.get('path', 'C:/Users/Сергей/ions_fizika/ions_ser/vs')
    selected_file = request.GET.get('file')
    ionogram_html = None
    noise_html = None
    passport = None
    error = None

    try:
        items = os.listdir(base_path)
        dirs = []
        dat_files = []

        for item in items:
            item_path = os.path.join(base_path, item)
            if os.path.isdir(item_path):
                dirs.append(item)
            elif item.endswith('.dat'):
                dat_files.append(item)
    except Exception as e:
        dirs = []
        dat_files = []
        error = str(e)

    if selected_file:
        file_path = os.path.join(base_path, selected_file)
        if os.path.isfile(file_path):
            try:
                ion = Ionogram()
                ion.readion(file_path)
                passport = ion.get_passport()
                ionogram_html = generate_ionogram(ion)
                noise_html = generate_noise_plot(ion)
            except Exception as e:
                error = f"Невозможно построить ионограмму: {str(e)}"

    parent_dir = os.path.dirname(base_path) if base_path != os.path.dirname(base_path) else None

    return render(request, 'ionograms/list.html', {
        'base_path': base_path,
        'parent_dir': parent_dir,
        'dirs': sorted(dirs),
        'dat_files': sorted(dat_files),
        'selected_file': selected_file,
        'ionogram_html': ionogram_html,
        'noise_html': noise_html,
        'passport': passport,
        'error': error
    })

def get_files_by_date(request):
    if request.method == 'GET' and 'date' in request.GET:
        date_str = request.GET['date']
        try:
            date = datetime.strptime(date_str, '%Y-%m-%d').date()
            month = f"{date.month:02d}"
            day = f"{date.day:02d}"

            all_dat_files = glob.glob('C:/Users/Сергей/ions_fizika/**/*.dat', recursive=True)

            matched_files = []
            for file_path in all_dat_files:
                filename = os.path.basename(file_path)
                if filename.count('_') >= 4:
                    try:
                        file_month, file_day, _, _, _ = filename.split('_')[:5]
                        if file_month == month and file_day == day:
                            matched_files.append({
                                'path': os.path.dirname(file_path),
                                'file': filename
                            })
                    except:
                        continue

            return JsonResponse({'files': matched_files})
        except ValueError:
            return JsonResponse({'error': 'Invalid date format'}, status=400)
    return JsonResponse({'error': 'Invalid request'}, status=400)

def generate_ionogram(ion):
    z = ion.get_ionogram()
    x = ion.get_frequences()
    y = ion.get_heights()
    path_value = ion._Ionogram__parameters['path']['value']
    mode_value = ion._Ionogram__parameters['mode']['value']
    date_time = ion.date_time

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            x=x, y=y, z=z,
            colorscale='jet',
            zmin=np.nanmin(z),
            zmax=np.nanmax(z),
            colorbar=dict(title='Amplitude, dB'),
            name='Ionogram',
            hovertemplate=(
                'Frequency: %{x:.2f} MHz<br>'
                'Range: %{y:.2f} km<br>'
                'Amplitude: %{z:.2f} dB<br>'
                '<extra></extra>'
            )
        )
    )

    fig.update_layout(
        title=f"{path_value}<br>{date_time.strftime('%d.%m.%Y %H:%M:%S')} UT",
        xaxis_title='frequency, MHz',
        yaxis_title='range, km' if mode_value == 'ВЗ' else 'Range, km',
        height=600,
        width=800,
        hovermode='closest',
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    fig.update_xaxes(
        range=[np.nanmin(x), 10],
        dtick=1,
        minor=dict(ticklen=4, dtick=0.2),
        showgrid=True,
        gridwidth=1,
        gridcolor='LightGrey'
    )
    fig.update_yaxes(
        range=[np.nanmin(y), np.nanmax(y)],
        dtick=100,
        minor=dict(ticklen=4, dtick=20),
        showgrid=True,
        gridwidth=1,
        gridcolor='LightGrey'
    )

    return fig.to_html(full_html=False)

def generate_noise_plot(ion):
    if not ion.noise:
        return None

    noise_data = np.array(ion.noise)
    frequencies = ion.get_frequences()[:-1]
    mask = frequencies <= 10
    frequencies = frequencies[mask]
    noise_levels = noise_data[mask, 1]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=frequencies,
            y=noise_levels,
            mode='markers',
            marker=dict(color='blue', size=3),
            hovertemplate=(
                'Frequency: %{x:.2f} MHz<br>'
                'Noise: %{y:.2f} dB<br>'
                '<extra></extra>'
            )
        )
    )

    fig.update_layout(
        xaxis_title='frequency, MHz',
        yaxis_title='noise, dB',
        height=300,
        width=800,
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    fig.update_xaxes(
        range=[np.nanmin(frequencies), 10],
        dtick=1,
        showgrid=True,
        gridcolor='LightGrey'
    )

    return fig.to_html(full_html=False)
