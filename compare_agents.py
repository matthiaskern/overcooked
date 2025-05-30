import argparse
import copy
import pandas as pd
import datetime
import time
from play import Player, TASKLIST

def run_with_configuration(config_name, ai_agent, human_agent, params, output_file=None, max_steps=300):
    player_params = copy.deepcopy(params)
    player_params['agent'] = ai_agent
    player_params['human'] = human_agent

    print(f"\n===== Running {config_name} =====")
    print(f"AI Agent: {ai_agent}, Human Agent: {human_agent}")

    player = Player(**player_params)
    data = player.run()

    rows = data[1:]
    total_steps = len(rows)
    completed = rows[-1][-1] if rows else False

    result = {
        'config_name': config_name,
        'ai_agent': ai_agent,
        'human_agent': human_agent,
        'completed': completed,
        'steps': total_steps,
        'data': rows
    }

    print(f"Results:")
    print(f"  Completed: {completed}")
    print(f"  Steps taken: {total_steps}")

    if output_file:
        columns = data[0]
        df = pd.DataFrame(rows, columns=columns)
        df.to_csv(output_file, index=False)
        print(f"Data saved to {output_file}")

    return result

def compare_configurations(configurations, trials=3, **params):
    all_results = []
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    task_name = TASKLIST[params['task']].replace(" ", "_").replace("-", "_")
    model_name = params.get('llm_model', '').replace("/", "_").replace(".", "_")

    for config_name, (ai_agent, human_agent) in configurations.items():
        config_results = []

        for trial in range(trials):
            print(f"\n===== {config_name} - Trial {trial+1}/{trials} =====")
            output_file = f"output_{config_name}_{task_name}_{timestamp}_trial{trial+1}.csv"

            result = run_with_configuration(
                config_name=config_name,
                ai_agent=ai_agent,
                human_agent=human_agent,
                params=params,
                output_file=output_file
            )

            config_results.append(result)

        completed_count = sum(1 for r in config_results if r['completed'])
        completion_rate = completed_count / trials
        avg_steps = sum(r['steps'] for r in config_results) / trials

        summary = {
            'config_name': config_name,
            'ai_agent': ai_agent,
            'human_agent': human_agent,
            'completion_rate': completion_rate,
            'completed_count': completed_count,
            'avg_steps': avg_steps,
            'trials': config_results
        }

        all_results.append(summary)

        print(f"\n===== {config_name} SUMMARY =====")
        print(f"Completion Rate: {completion_rate*100:.1f}% ({completed_count}/{trials})")
        print(f"Average Steps: {avg_steps:.1f}")

    comparison_data = []
    for result in all_results:
        comparison_data.append({
            'Configuration': result['config_name'],
            'AI Agent': result['ai_agent'],
            'Human Agent': result['human_agent'],
            'Horizon Length': params.get('horizon_length', 3),
            'Completion Rate': f"{result['completion_rate']*100:.1f}%",
            'Avg Steps': f"{result['avg_steps']:.1f}",
        })

    comparison_df = pd.DataFrame(comparison_data)

    config_list = '_'.join(sorted(configurations.keys()))
    horizon_info = f"h{params.get('horizon_length', 3)}"
    summary_filename = f"agent_comparison_{config_list}_{task_name}_{model_name}_{horizon_info}_{timestamp}.csv"
    comparison_df.to_csv(summary_filename, index=False)

    print(f"\n===== FINAL COMPARISON (Horizon Length: {params.get('horizon_length', 3)}) =====")
    print(f"Summary saved to: {summary_filename}")
    print(comparison_df.to_string(index=False))

    return all_results, comparison_df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--grid_dim', type=int, nargs=2, default=[5, 5], help='Grid world size')
    parser.add_argument('--task', type=int, default=6, help='The recipe agent cooks')
    parser.add_argument('--map_type', type=str, default="A", help='The type of map')
    parser.add_argument('--mode', type=str, default="vector", help='The type of observation (vector/image)')
    parser.add_argument('--debug', action='store_true', help='Whether to print debug information and render')
    parser.add_argument('--llm_model', type=str, default="openai/gpt-4.1", help='LLM model to use')
    parser.add_argument('--horizon_lengths', type=int, nargs='+', default=[3], help='Set the planning horizon lengths to compare')
    parser.add_argument('--trials', type=int, default=3, help='Number of trials per configuration')
    parser.add_argument('--configs', type=str, nargs='+', default=['llm_vs_stationary', 'llm_vs_random'],
                      help='Configurations to compare')

    args = parser.parse_args()

    all_configurations = {
        'llm_vs_stationary': ('llm', 'stationary'),
        'llm_vs_random': ('llm', 'random'),
        'llm_vs_llm': ('llm', 'llm'),
        'multimodal_vs_stationary': ('multimodal', 'stationary'),
        'multimodal_vs_random': ('multimodal', 'random'),
        'multimodal_vs_llm': ('multimodal', 'llm'),
        'multimodal_vs_multimodal': ('multimodal', 'multimodal'),
        'multimodal_vs_human': ('multimodal', 'human'),
    }

    configurations_to_run = {}
    for config_name in args.configs:
        if config_name in all_configurations:
            configurations_to_run[config_name] = all_configurations[config_name]
        else:
            print(f"Warning: Unknown configuration '{config_name}'. Skipping.")

    if not configurations_to_run:
        print("Error: No valid configurations specified!")
        exit(1)

    all_results = []
    all_comparisons = []

    for horizon_length in args.horizon_lengths:
        base_params = {
            'grid_dim': args.grid_dim,
            'task': args.task,
            'map_type': args.map_type,
            'mode': args.mode,
            'debug': args.debug,
            'llm_model': args.llm_model,
            'horizon_length': horizon_length
        }

        print(f"\n===== RUNNING WITH HORIZON LENGTH {horizon_length} =====")
        print(f"Comparing {len(configurations_to_run)} configurations with {args.trials} trials each:")
        for config_name, (ai_agent, human_agent) in configurations_to_run.items():
            print(f"  - {config_name}: AI={ai_agent}, Human={human_agent}")
        print(f"Task: {TASKLIST[args.task]}")

        results, comparison = compare_configurations(
            configurations=configurations_to_run,
            trials=args.trials,
            **base_params
        )

        all_results.append(results)
        all_comparisons.append(comparison)

    if len(args.horizon_lengths) > 1:
        print("\n===== COMBINED RESULTS ACROSS ALL HORIZON LENGTHS =====")
        combined_df = pd.concat(all_comparisons)
        combined_filename = f"agent_comparison_combined_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        combined_df.to_csv(combined_filename, index=False)
        print(f"Combined summary saved to: {combined_filename}")
        print(combined_df.to_string(index=False))
