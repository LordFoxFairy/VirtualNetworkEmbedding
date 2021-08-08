import os, sys

current_path = os.path.dirname(os.path.realpath(__file__))
PROJECT_HOME = os.path.abspath(os.path.join(current_path, os.pardir))
if PROJECT_HOME not in sys.path:
    sys.path.append(PROJECT_HOME)

from main.common_main import *




agents = [agent]
agent_labels = [config.TARGET_ALGORITHM.value]

performance_revenue = np.zeros(shape=(1, config.GLOBAL_MAX_STEPS + 1))
performance_acceptance_ratio = np.zeros(shape=(1, config.GLOBAL_MAX_STEPS + 1))
performance_rc_ratio = np.zeros(shape=(1, config.GLOBAL_MAX_STEPS + 1))
performance_link_fail_ratio = np.zeros(shape=(1, config.GLOBAL_MAX_STEPS + 1))


def main():
    start_ts = time.time()
    for run in range(config.NUM_RUNS):
        run_start_ts = time.time()

        env = VNEEnvironment(logger)
        envs = [env]

        utils.print_env_and_agent_info(env, agents[0], logger)

        agent_id = 0

        states = [envs[agent_id].reset()]

        done = False
        time_step = 0

        while not done:
            time_step += 1

            before_action_msg = "state {0} | ".format(repr(states[agent_id]))
            before_action_simple_msg = "state {0} | ".format(states[agent_id])
            logger.info("{0} {1}".format(
                utils.run_agent_step_prefix(run + 1, agent_id, time_step), before_action_msg
            ))

            # action = bl_agent.get_action(state)
            action = agents[agent_id].get_action(states[agent_id])

            action_msg = "act. {0:30} |".format(
                str(action) if action.vnrs_embedding is not None and action.vnrs_postponement is not None else " - "
            )
            logger.info("{0} {1}".format(
                utils.run_agent_step_prefix(run + 1, agent_id, time_step), action_msg
            ))

            next_state, reward, done, info = envs[agent_id].step(action)

            elapsed_time = time.time() - run_start_ts
            after_action_msg = "reward {0:6.1f} | revenue {1:6.1f} | acc. ratio {2:4.2f} | " \
                               "r/c ratio {3:4.2f} | {4}".format(
                reward, info['revenue'], info['acceptance_ratio'], info['rc_ratio'],
                time.strftime("%Hh %Mm %Ss", time.gmtime(elapsed_time)),
            )

            after_action_msg += " | {0:3.1f} steps/sec.".format(time_step / elapsed_time)

            logger.info("{0} {1}".format(
                utils.run_agent_step_prefix(run + 1, agent_id, time_step), after_action_msg
            ))

            print("{0} {1} {2} {3}".format(
                utils.run_agent_step_prefix(run + 1, agent_id, time_step),
                before_action_simple_msg,
                action_msg,
                after_action_msg
            ))

            states[agent_id] = next_state

            performance_revenue[agent_id, time_step] += info['revenue']
            performance_acceptance_ratio[agent_id, time_step] += info['acceptance_ratio']
            performance_rc_ratio[agent_id, time_step] += info['rc_ratio']
            performance_link_fail_ratio[agent_id, time_step] += \
                info['link_embedding_fails_against_total_fails_ratio']

            logger.info("")

            if time_step > config.FIGURE_START_TIME_STEP - 1 and time_step % 100 == 0:
                draw_performance(
                    agents=agents, agent_labels=agent_labels, run=run, time_step=time_step,
                    performance_revenue=performance_revenue / (run + 1),
                    performance_acceptance_ratio=performance_acceptance_ratio / (run + 1),
                    performance_rc_ratio=performance_rc_ratio / (run + 1),
                    performance_link_fail_ratio=performance_link_fail_ratio / (run + 1),
                )

        draw_performance(
            agents=agents, agent_labels=agent_labels, run=run, time_step=time_step,
            performance_revenue=performance_revenue / (run + 1),
            performance_acceptance_ratio=performance_acceptance_ratio / (run + 1),
            performance_rc_ratio=performance_rc_ratio / (run + 1),
            performance_link_fail_ratio=performance_link_fail_ratio / (run + 1),
            send_image_to_slack=True
        )

        msg = "RUN: {0} FINISHED - ELAPSED TIME: {1}".format(
            run + 1, time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_ts))
        )
        logger.info(msg), print(msg)


if __name__ == "__main__":
    main()