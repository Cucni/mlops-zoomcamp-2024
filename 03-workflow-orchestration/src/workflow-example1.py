from prefect import flow, task


@task
def my_task(x):
    # This is a python function that acts as a prefect task, i.e. a unit of work
    print(x)


@flow
def my_flow(x):
    # This is a python function that acts as a prefect flow, i.e. a logical container
    # for workflow logic. In this case the flow triggers a task
    my_task(x)


if __name__ == "__main__":
    my_flow(10)
    my_flow(20)
