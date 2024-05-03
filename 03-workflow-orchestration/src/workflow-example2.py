from prefect import task, flow


@task(log_prints=True)
def my_task2(n):
    # python function acting as a prefect task, i.e. a unit of work
    print(n**2)


@flow
def my_flow2(n):
    # python function acting as a prefect flow, i.e. container for workflow logic
    # in this example in a single flow run we call two tasks, which are actually the
    # same task with different parameters
    my_task2(n)
    my_task2(n + 1)


if __name__ == "__main__":
    my_flow2(1)
    my_flow2(3)
