import configparser
config = configparser.ConfigParser()

base_file = '../scale.cfg'

def replace_llc_dataflow_single(llc, dataflow):
    run_name = "alexnet_c"
    run_name += llc
    run_name += '_'
    run_name += dataflow

    default_config = configparser.RawConfigParser()
    default_config.optionxform = str
    default_config.read(base_file)
    new_config = default_config

    if llc.isnumeric():
        new_config['llc']['SizekB'] = llc
    else:
        if llc == 'idealLLC+realMem':
            new_config['llc']['AlwaysHit'] = '1'
        elif llc == 'idealLLC+idealMem':
            new_config['llc']['Bypassing'] = '1'
            # new_config['llc']['HitLatency'] = '1'
        elif llc == 'noLLC':
            run_name = "alexnet_"
            run_name += llc
            run_name += '_'
            run_name += dataflow

            new_config['llc']['HitLatency'] = '40'
            
    new_config['general']['run_name'] = run_name
    new_config['architecture_presets']['BatchSize'] = '1'
    new_config['architecture_presets']['Dataflow'] = dataflow
    new_config['architecture_presets']['Unified'] = '1'

    file_name = run_name + '.cfg'

    with open(file_name, 'w') as configfile:
        new_config.write(configfile)
        
        
def replace_llc_dataflow(num_pe, num_batch, llc, dataflow):
    run_name = "alexnet_c"
    run_name += llc
    run_name += '_'
    run_name += str(num_pe)
    run_name += '_'
    run_name += str(num_batch)
    run_name += '_'
    run_name += dataflow

    default_config = configparser.RawConfigParser()
    default_config.optionxform = str
    default_config.read(base_file)
    new_config = default_config

    new_config['llc']['SizekB'] = llc

    new_config['general']['run_name'] = run_name
    # new_config['architecture_presets']['NumPE'] = str(num_pe)
    new_config['architecture_presets']['BatchSize'] = str(num_batch * int(num_pe))
    new_config['architecture_presets']['Dataflow'] = dataflow
    new_config['architecture_presets']['Unified'] = '0'

    new_config['architecture_presets']['ArrayHeight'] = str(int(default_config['architecture_presets']['ArrayHeight']) * int(num_pe))
    new_config['architecture_presets']['ArrayWidth'] = str(int(default_config['architecture_presets']['ArrayWidth']) * int(num_pe))
    new_config['architecture_presets']['Bandwidth'] = str(int(default_config['architecture_presets']['Bandwidth']) * int(num_pe))

    new_config['architecture_presets']['IfmapSramSzkB'] = str(int(default_config['architecture_presets']['IfmapSramSzkB']) * int(pow(num_pe, 2)))
    new_config['architecture_presets']['FilterSramSzkB'] = str(int(default_config['architecture_presets']['FilterSramSzkB']) * int(pow(num_pe, 2)))
    new_config['architecture_presets']['OfmapSramSzkB'] = str(int(default_config['architecture_presets']['OfmapSramSzkB']) * int(pow(num_pe, 2)))
    new_config['llc']['SizekB'] = str(int(default_config['llc']['SizekB']) * int(num_pe))

    file_name = run_name + '.cfg'

    with open(file_name, 'w') as configfile:
        new_config.write(configfile)
        
        
def replace_llc_assoc(num_pe, num_batch, llc, dataflow):
    run_name = "alexnet_c"
    run_name += llc
    run_name += '_'
    run_name += str(num_pe)
    run_name += '_'
    run_name += str(num_batch)
    run_name += '_'
    run_name += dataflow

    default_config = configparser.RawConfigParser()
    default_config.optionxform = str
    default_config.read(base_file)
    new_config = default_config

    new_config['llc']['SizekB'] = llc

    new_config['general']['run_name'] = run_name
    # new_config['architecture_presets']['NumPE'] = str(num_pe)
    new_config['architecture_presets']['BatchSize'] = str(num_batch)
    new_config['architecture_presets']['Dataflow'] = dataflow
    new_config['architecture_presets']['Unified'] = '0'

    new_config['llc']['SizekB'] = '1024'
    new_config['llc']['Assoc'] = llc
    new_config['llc']['Partition'] = str(int(pow(2, int(llc))))

    file_name = run_name + '.cfg'

    with open(file_name, 'w') as configfile:
        new_config.write(configfile)


def replace_llc_mmap_order(num_pe, num_batch, llc, dataflow):
    run_name = "alexnet_c"
    run_name += llc
    run_name += '_'
    run_name += str(num_pe)
    run_name += '_'
    run_name += str(num_batch)
    run_name += '_lm'

    default_config = configparser.RawConfigParser()
    default_config.optionxform = str
    default_config.read(base_file)
    new_config = default_config

    new_config['llc']['SizekB'] = llc

    new_config['general']['run_name'] = run_name
    new_config['architecture_presets']['BatchSize'] = str(num_batch * int(num_pe))
    new_config['architecture_presets']['Dataflow'] = dataflow
    new_config['architecture_presets']['Unified'] = '0'

    new_config['architecture_presets']['ArrayHeight'] = str(int(default_config['architecture_presets']['ArrayHeight']) * int(num_pe))
    new_config['architecture_presets']['ArrayWidth'] = str(int(default_config['architecture_presets']['ArrayWidth']) * int(num_pe))
    new_config['architecture_presets']['Bandwidth'] = str(int(default_config['architecture_presets']['Bandwidth']) * int(num_pe))

    new_config['architecture_presets']['IfmapSramSzkB'] = str(int(default_config['architecture_presets']['IfmapSramSzkB']) * int(pow(num_pe, 2)))
    new_config['architecture_presets']['FilterSramSzkB'] = str(int(default_config['architecture_presets']['FilterSramSzkB']) * int(pow(num_pe, 2)))
    new_config['architecture_presets']['OfmapSramSzkB'] = str(int(default_config['architecture_presets']['OfmapSramSzkB']) * int(pow(num_pe, 2)))
    new_config['architecture_presets']['TensorMainOrder'] = '0'
    
    new_config['llc']['SizekB'] = str(int(default_config['llc']['SizekB']) * int(num_pe))

    file_name = run_name + '.cfg'

    with open(file_name, 'w') as configfile:
        new_config.write(configfile)


def llc_dataflow():
    
    dataflow_list = ['is', 'os', 'ws']
    llc_size_list = ['256', '512', '1024', '2048', '4096', '8192', '16384', '32768', '65536', 'idealLLC+realMem', 'idealLLC+idealMem', 'noLLC']
    
    for llc in llc_size_list:
        for dataflow in dataflow_list:
            replace_llc_dataflow_single(llc, dataflow)
            
            
    dataflow_list = ['ws']
    llc_size_list = ['256', '512', '1024', '2048', '4096']
    
    num_pe_list = [1]
    num_batch_list = [1, 2, 4, 8, 16, 32]
    for num_pe in num_pe_list:
        for num_batch in num_batch_list:
            for llc in llc_size_list:
                for dataflow in dataflow_list:
                    replace_llc_dataflow(num_pe, num_batch, llc, dataflow)

                    
    num_pe_list = [1, 2, 4, 8]
    num_batch_list = [1]
    for num_pe in num_pe_list:
        for num_batch in num_batch_list:
            for llc in llc_size_list:
                for dataflow in dataflow_list:
                    replace_llc_dataflow(num_pe, num_batch, llc, dataflow)

                    
    num_pe_list = [1]
    num_batch_list = [1]
    llc_size_list = ['1', '2', '3', '4']
    for num_pe in num_pe_list:
        for num_batch in num_batch_list:
            for llc in llc_size_list:
                for dataflow in dataflow_list:
                    replace_llc_assoc(num_pe, num_batch, llc, dataflow)
              
    
    dataflow_list = ['ws']
    llc_size_list = ['256', '512', '1024', '2048', '4096']
    num_pe_list = [1]
    num_batch_list = [1]
    for num_pe in num_pe_list:
        for num_batch in num_batch_list:
            for llc in llc_size_list:
                for dataflow in dataflow_list:
                    replace_llc_mmap_order(num_pe, num_batch, llc, dataflow)            

def main():
   llc_dataflow() 

if __name__ == '__main__':
    main()
