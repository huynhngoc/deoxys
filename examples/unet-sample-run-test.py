from deoxys.experiment import Experiment

exp = Experiment(
    log_base_path='../../hn_perf/test'
).from_file(
    '../../hn_perf/logs_saved/model.003.edited.h5'
).run_test(
    masked_images=[i for i in range(42)]
).run_test(
    masked_images=[i for i in range(198)],
    use_original_image=True,
    image_name='Patient{patient_idx:03d}_Slice{slice_idx:03d}',
    image_title_name='Patient {patient_idx:03d}, Slice {slice_idx:03d}'
)
