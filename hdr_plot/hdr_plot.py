#
# hdr-plot.py v0.2.3 - A simple HdrHistogram plotting script.
# Copyright © 2018 Bruno Bonacci - Distributed under the Apache License v 2.0
#
# usage: hdr-plot.py [-h] [--output OUTPUT] [--title TITLE] [--legend_fontsize FONTSIZE] [--nobox] files [files ...]
#
# A standalone plotting script for https://github.com/giltene/wrk2 and
#  https://github.com/HdrHistogram/HdrHistogram.
#
# This is just a quick and unsophisticated script to quickly plot the
# HdrHistograms directly from the output of `wkr2` benchmarks.
#
#
import argparse
import re
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import numpy as np

# global settings
fontsize=20


#
# parsing and plotting functions
#

regex = re.compile(r'\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)')
filename = re.compile(r'(.*/)?([^.]*)(\.\w+\d+)?')

regex_lines = re.compile(r'(.+((requests)|(99\.000%)).+)|(#)|(Req)')
regex_val = re.compile(r'requests=([0-9]+)')
regex_thr = re.compile(r'\s+([0-9]+)\s+requests')
regex_lat = re.compile(r'.+(99\.000%)+\s+([0-9.\w]+)')

def parse_percentiles( file ):
    lines       = [ line for line in open(file) if re.match(regex, line)]
    values      = [ re.findall(regex, line)[0] for line in lines]
    pctles      = [ (float(v[0]), float(v[1]), int(v[2]), float(v[3])) for v in values]
    percentiles = pd.DataFrame(pctles, columns=['Latency', 'Percentile', 'TotalCount', 'inv-pct'])
    return percentiles


def parse_files( files ):
    return [ parse_percentiles(file) for file in files]


def info_text(name, data):
    textstr = '%-18s\n------------------\n%-6s = %6.2f ms\n%-6s = %6.2f ms\n%-6s = %6.2f ms\n'%(
        name,
        "min",    data['Latency'].min(),
        "median", data.iloc[(data['Percentile'] - 0.5).abs().argsort()[:1]]['Latency'],
        "max",    data['Latency'].max())
    return textstr


def info_box(ax, text):
    props = dict(boxstyle='round', facecolor='lightcyan', alpha=0.5)

    # place a text box in upper left in axes coords
    ax.text(0.05, 0.95, text, transform=ax.transAxes,
        verticalalignment='top', bbox=props, fontname='monospace')


def plot_summarybox( ax, percentiles, labels ):
    # add info box to the side
    textstr = '\n'.join([info_text(labels[i], percentiles[i]) for i in range(len(labels))])
    info_box(ax, textstr)


def plot_percentiles( percentiles, labels, legend_fontsize=fontsize):
    fig, ax = plt.subplots(figsize=(16,8))
    # plot values
    for data in percentiles:
        ax.plot(data['Percentile'], data['Latency'])

    # set axis and legend
    ax.grid()
    ax.set_xlabel('Percentile', fontsize=fontsize)
    ax.set_ylabel('Latency (milliseconds)', fontsize=fontsize)
    ax.set_title('Latency Percentiles (lower is better)', fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    ax.set_xscale('logit')
    ax.get_xticklabels()[1].set_color("red")
    plt.xticks([0.9, 0.99, 0.999, 0.9999], fontsize=fontsize)
    majors = ["90%", "99%", "99.9%", "99.99%", "99.999%", "99.9999%"]

    ax.xaxis.set_major_formatter(ticker.FixedFormatter(majors))
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    plt.legend(labels, loc='upper left', fontsize=legend_fontsize)

    return fig, ax

def plot_thr_req(tail_data, labels, legend_location,legend_fontsize=fontsize):
    # general setup
    fig, ax = plt.subplots(figsize=(16,8))

    # create new relevant labels
    data = pd.concat(tail_data)
    data['filename'] = labels
    data = data.sort_values(by='Requests').reset_index(drop=True)

    # labels = data['Orchestrator'].unique() 
    labels = ["swarm","k8s", "nomad"]
    lines = ["-","--","-.",":"]
    markers = ["o","v","^","<",">","8","s","p","*","h","H","D","d","P","X"]
    orchestrator = data['Orchestrator'].unique()[0]
    benchmark = data['Benchmark'].unique()[0]

    for mark, orch in enumerate(labels):
        plot_data = data.loc[data['Orchestrator'] == orch]
        ax.plot(plot_data['Requests'], plot_data['Throughput'], label=f"{benchmark}-{orch}-baseline", linestyle=lines[mark % len(markers)])

    # set axis and legend
    ax.grid()
    ax.set_xlim(left=0)
    ax.set_xlabel('Requests', fontsize=fontsize)
    ax.set_ylabel('Throughput', fontsize=fontsize)
    ax.set_title('Throughput versus Requests', fontsize=fontsize)

    xticks = [int(tick) for tick in ax.get_xticks()]
    plt.yticks(fontsize=fontsize)
    plt.xticks(xticks, fontsize=fontsize)
    plt.legend(loc=legend_location, fontsize=legend_fontsize)

    return fig, ax


def plot_lat_req_all(tail_data, labels, legend_location,legend_fontsize=fontsize):
    # general setup
    fig, ax = plt.subplots(figsize=(16,8))

    # create new relevant labels
    data = pd.concat(tail_data)
    data['filename'] = labels
    data = data.sort_values(by='Requests').reset_index(drop=True)

    # labels
    labels = ["swarm","k8s", "nomad"]
    lines = ["-","--","-.",":"]
    markers = ["o","v","^","<",">","8","s","p","*","h","H","D","d","P","X"]
    benchmark = data['Benchmark'].unique()[0]

    print("Multiple orchestrators one benchmark and one param")
    for mark, orch in enumerate(labels):
        plot_data = data.loc[data['Orchestrator'] == orch]
        vertical = plot_data['vertical'].unique()[0]
        horizontal = plot_data['horizontal'].unique()[0]
        availability = plot_data['availability'].unique()[0]
        baseline = plot_data['baseline'].unique()[0]
        if vertical == 0:
            print("vertical")
            ax.plot(plot_data['Requests'], plot_data['Latency'], label=f"{benchmark}-{orch}-vertical", linestyle=lines[mark % len(markers)])
        elif  horizontal == 0:
            print("horizontal")
            ax.plot(plot_data['Requests'], plot_data['Latency'], label=f"{benchmark}-{orch}-horizontal", linestyle=lines[mark % len(markers)])
        elif availability == 1 and baseline == 1:
            print("availability")
            ax.plot(plot_data['Requests'], plot_data['Latency'], label=f"{benchmark}-{orch}-availability", linestyle=lines[mark % len(markers)])
        else:
            print("baseline")
            ax.plot(plot_data['Requests'], plot_data['Latency'], label=f"{benchmark}-{orch}-baseline", linestyle=lines[mark % len(markers)])

    # set axis and legend
    ax.grid()
    ax.set_xlim(left=0)
    ax.set_xlabel('Requests', fontsize=fontsize)
    ax.set_ylabel('Latency (milliseconds)', fontsize=fontsize)
    ax.set_title('Latency versus Requests', fontsize=fontsize)

    xticks = [int(tick) for tick in ax.get_xticks()]
    plt.yticks(fontsize=fontsize)
    plt.xticks(xticks, fontsize=fontsize)
    plt.legend(loc=legend_location, fontsize=legend_fontsize)

    return fig, ax
    

def plot_lat_req(data, labels, legend_location,legend_fontsize=fontsize):
    # general setup
    fig, ax = plt.subplots(figsize=(16,8))


    labels = data['Benchmark'].unique() 
    lines = ["-","--","-.",":"]
    markers = ["o","v","^","<",">","8","s","p","*","h","H","D","d","P","X"]
    orchestrator = data['Orchestrator'].unique()[0]

    # one benchmark but multiple params
    if len(labels) == 1:
        print("one benchmark but multiple params")
        for benchmark in labels:
            for mark, param in enumerate(['Horizontal', 'Vertical', 'Availability','Baseline']):
                plot_data = data.loc[data['Benchmark'] == benchmark]
                if len(plot_data[param].unique()) == 0:
                    continue

                if param == 'Availability':
                    plot_data = plot_data.loc[plot_data[param] == 1]
                    if plot_data[param].unique() == 1:
                        ax.plot(plot_data['Requests'], plot_data['Latency'], label=f"{benchmark}-{orchestrator}-{param.lower}", linestyle=lines[mark % len(markers)])
                else:
                    plot_data = plot_data.loc[plot_data[param] == 0]
                    if plot_data[param].unique() == 0:
                        ax.plot(plot_data['Requests'], plot_data['Latency'], label=f"{benchmark}-{orchestrator}-{param}", linestyle=lines[mark % len(markers)])
    # multiple benchmarks but one params
    else:
        print("Multiple benchmarks but one param")
        for mark, benchmark in enumerate(labels):
                plot_data = data.loc[data['Benchmark'] == benchmark]
                ax.plot(plot_data['Requests'], plot_data['Latency'], label=f"{benchmark}-{orchestrator}", linestyle=lines[mark % len(markers)])
    # ax.plot(data['Requests'], data['Latency'])

    
    # set axis and legend
    ax.grid()
    # sb.despine(ax=ax, offset=0)
    ax.set_xlim(left=0)
    ax.set_xlabel('Requests', fontsize=fontsize)
    ax.set_ylabel('Latency (milliseconds)', fontsize=fontsize)
    ax.set_title('Latency versus Requests', fontsize=fontsize)

    xticks = [int(tick) for tick in ax.get_xticks()]
    # print(xticks)
    # yticks = [int(tick) for tick in ax.get_yticks()]
    # print(xticks, yticks)
    # plt.yticks(yticks, fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    # plt.xticks(xticks, fontsize=fontsize)
    plt.xticks(xticks, fontsize=fontsize)
    plt.legend(loc=legend_location, fontsize=legend_fontsize)

    return fig, ax



def save_df(df, filename="default.csv"):
    df.to_csv(filename, sep=',', index=False)

def parse_files_tail(files):
    df = pd.concat([ parse_prelim(file) for file in files]).sort_values(by='Requests').reset_index(drop=True)
    return df


def parse_prelim(file):
    # swarm-sn-wrk-mixed-c6525-25g-exp0-havail0-hori1-verti1-infi0-t4-c8-d30-R200
    # params we need to derive from the file
    lines = [ line for line in open(file) if re.match(regex_lines, line) ]
    # print(lines)
    requests  = float([ val for val in  [ re.findall(regex_val, line) for line in lines ] if len(val) != 0 ][0][0])
    latency  = [ val for val in [ re.findall(regex_lat, line) for line in lines ][1] if len(val) != 0 ][0][1]
    if latency[-2] == 'm':
        latency = float(latency[:-2])
    else:
        latency = float(latency[:-1]) * 1000
    total_throughput  = float([ val for val in  [ re.findall(regex_thr, line) for line in lines ] if len(val) != 0 ][0][0])
    regex_thr_measured = ".+(Total count\s+=\s+)([0-9]+)"
    measured_throughput  = [ re.search(regex_thr_measured, line) for line in lines if re.search(regex_thr_measured, line) ][0].group(2)
    regex_mean= "#\[Mean\s+=\s+(\w+.\w+)"
    mean  = float([ re.search(regex_mean, line) for line in lines if re.search(regex_mean, line) ][0].group(1))
    regex_max= "#\[Max\s+=\s+(\w+.\w+)"
    max_val = float([ re.search(regex_max, line) for line in lines if re.search(regex_max, line) ][0].group(1))
    regex_stddev= ".+Std\w+\s+=\s+(\w+\.\w+)"
    stddev  = float([ re.search(regex_stddev, line) for line in lines if re.search(regex_stddev, line) ][0].group(1))
    regex_reqsec= "Requests/.+\W(\w+\.\w+)"
    reqsec  = float([ re.search(regex_reqsec, line) for line in lines if re.search(regex_reqsec, line) ][0].group(1))
    # print(requests, latency, throughput, mean, max_val, stddev, reqsec)

    # params stated in the filename
    orchestrator =  re.search('(swarm|k8s|nomad)', file).group(1)
    benchmark = re.search('(sn|hr|mm)', file).group(1)
    infinite = int(re.search('(infi)(1|0)', file).group(2))
    exp = re.search('(exp[0-9]+)', file).group(1)
    availability = int(re.search('(havail)(0|1)', file).group(2))
    horizontal = int(re.search('(hori)(1|0)', file).group(2))
    N = int(re.search('(N)(.)', file).group(1))
    vertical = int(re.search('(verti)(1|0)', file).group(2))
    threads = int(re.search('(t)([0-9]{1,2})-c', file).group(2))
    connections = int(re.search('c([0-9]{1,4})-d', file).group(1))
    duration = int(re.search('d([0-9]+)', file).group(1))
    if int(availability) == 0 and int(horizontal) == 1 and int(vertical) == 1: 
        baseline = 0
    else:
        baseline = 1
    # print(orchestrator, benchmark, infinite, exp, availability, horizontal, vertical, threads, connections, duration, latency, throughput, mean, max_val, stddev, reqsec, baseline)
    # print("orchestrator", "benchmark", "infinite", "exp", "availability", "horizontal", "vertical", "threads", "connections", "duration", "latency", "throughput", "mean", "max", "stddev", "reqsec", "baseline")
    

# df = pd.DataFrame(zip(latency, requests, throughput, benchmark, orchestrator),
    df = pd.DataFrame({'Latency': [latency],
        'Requests': [requests], 
        'Throughput': [total_throughput],
        'Measured_Throughput': [measured_throughput],
        'Baseline': [baseline],
        'Orchestrator': [orchestrator],
        'Benchmark': [benchmark],
        'Infinite': [infinite],
        'Exp': [exp],
        'Availability': [availability],
        'Horizontal': [horizontal],
        'Vertical': [vertical],
        'Threads': [threads],
        'Connections': [connections],
        'Duration': [duration],
        'Mean': [mean],
        'Max': [max_val],
        'StdDev': [stddev],
        'ReqSec': [reqsec],
        'N': [N]}
    )  
    # print(df.head())
    return df


def arg_parse():
    parser = argparse.ArgumentParser(description='Plot HDRHistogram latencies.')
    parser.add_argument('files', nargs='+', help='list HDR files to plot')
    parser.add_argument('--output', default='latency.png',
                        help='Output file name (default: latency.png)')
    parser.add_argument('--title', default='', help='The plot title.')
    parser.add_argument("--nobox", help="Do not plot summary box", action="store_true")
    parser.add_argument('--lat_req', help="Plot latency vs requests", action="store_true")
    parser.add_argument('--thr_req', help="Plot throughput vs requests", action="store_true")
    parser.add_argument('--lat_req_all', help="Plot latency vs requests all orchestrators baseline", action="store_true")
    parser.add_argument('--save', help="Save the dataframe", action="store_true")
    parser.add_argument('--head', help="Print the head of the dataframe", action="store_true")
    parser.add_argument('--legend_location', default="lower right", help="Change the legend location")
    parser.add_argument("--legend_fontsize", default=20, help="Change the legend fontsize")
    args = parser.parse_args()
    return args


def main():
    # print command line arguments
    args = arg_parse()
    labels = [re.findall(filename, file)[0][1] for file in args.files]
    legend_fontsize = int(args.legend_fontsize)
    legend_location = args.legend_location

    # load the data and create the plot
    # print(args.files)
    pct_data = parse_files(args.files)
    tail_data = parse_files_tail(args.files)


    if args.head:
        print(tail_data.head())

    if args.save:
        name = tail_data['Exp'].unique()[0]
        # print(name[-2:])
        if name[-2:] == '12' or name[-2:] == '13' or name[-2:] == '14' or name[-2:] == '15':
            orch = tail_data['Orchestrator'].unique()[0]
            save_df(tail_data, f'df-{name}-{orch}.csv')
        else:
            save_df(tail_data, f'df-{name}.csv')
        return 0

    # plotting data
    if args.lat_req:
        fig, ax = plot_lat_req(tail_data, labels, legend_location, legend_fontsize)
    elif args.lat_req_all:
        fig, ax = plot_lat_req_all(tail_data, labels, legend_location, legend_fontsize)
    elif args.thr_req:
        fig, ax = plot_thr_req(tail_data, labels, legend_location, legend_fontsize)
    else:
        fig, ax = plot_percentiles(pct_data, labels, legend_fontsize)

    # plotting summary box
    if not args.nobox:
        plot_summarybox(ax, pct_data, labels)

    # add title
    plt.suptitle(args.title)
    
    # save image
    plt.savefig(args.output)
    print( "Wrote: " + args.output)


# for testing
if __name__ == "__main__":
    main()


