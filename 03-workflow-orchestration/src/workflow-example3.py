from prefect import task, flow


@task(log_prints=True)
def my_task3(n):
    # python function acting as a prefect task, i.e. a unit of work
    print(n**2)


@flow
def my_subflow(n):
    # python function acting as a prefect flow, but which is also called inside another
    # flow. This is a subflow, and will run tasks inside itself. This also establishes
    # a dependency between the main flow and this subflow.
    my_task3(10 * n)


@flow
def my_flow3(n):
    # python function acting as a prefect flow, i.e. container for workflow logic
    # in this example in a single flow run we call a task and a (sub)flow
    my_task3(n)
    my_subflow(n)


if __name__ == "__main__":
    my_flow3(1)
    my_flow3(3)
