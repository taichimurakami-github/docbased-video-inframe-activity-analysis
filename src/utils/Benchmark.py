from datetime import datetime


class Benchmark:
    # private _benchmark_laptime_queue =
    # format -> [key: str] : [(Done or Undone), startTime: datetime, endTime: datetime, diff: timedelta]
    # {
    #     "_lap_id_01_": [True, 23493939, 23498323, 4384],
    #     "_lap_id_02_": [False, 23593821],
    #     ...
    # }
    _benchmark_laptime_queue = {}

    def start_benchmark_lap(self, lap_id: str):
        if lap_id in self._benchmark_laptime_queue.keys():
            raise Exception(f"ERROR: Duplicated lap_id({lap_id}) detected")
        self._benchmark_laptime_queue[lap_id] = [False, datetime.now()]

    def end_benchmark_lap(self, lap_id: str):
        self._benchmark_laptime_queue[lap_id].append(datetime.now())
        self._benchmark_laptime_queue[lap_id][0] = True  # set status to Done

    def show_benchmark_results(self):
        for lap_id, laptime_data in self._benchmark_laptime_queue.items():
            status = laptime_data[0]
            t_start = laptime_data[1]
            t_end = None
            t_delta = None

            if status is not None:
                t_end = laptime_data[2]
                t_delta = t_end - t_start
                print(
                    f"[{lap_id}] start at {t_start} , end at {t_end} , delta = {t_delta}"
                )

            else:
                print(
                    f"[{lap_id}] start at {t_start} , This lap has never finished"
                )