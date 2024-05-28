path_prefix=/data1/users/itseyes/23snn_ctag/
target_path="./ana_0321/"
mkdir -p $target_path/log

root -l -b -q 'ana_ttbar.C("'$path_prefix'/tthbb+/Events/run_01_decayed_1/tag_1_delphes_events.root", "'$target_path'/tthbb+.root")'  &> ./$target_path/log/log00  &
root -l -b -q 'ana_ttbar.C("'$path_prefix'/tthbb-/Events/run_01_decayed_1/tag_1_delphes_events.root", "'$target_path'/tthbb-.root")'  &> ./$target_path/log/log01  &

root -l -b -q 'ana_ttbar.C("'$path_prefix'/ttbb+/Events/run_01_decayed_1/tag_1_delphes_events.root", "'$target_path'/ttbb+.root")'  &> ./$target_path/log/log20  &
root -l -b -q 'ana_ttbar.C("'$path_prefix'/ttbb-/Events/run_01_decayed_1/tag_1_delphes_events.root", "'$target_path'/ttbb-.root")'  &> ./$target_path/log/log21  &

root -l -b -q 'ana_ttbar.C("'$path_prefix'/ttcc+/Events/run_01_decayed_1/tag_1_delphes_events.root", "'$target_path'/ttcc+.root")'  &> ./$target_path/log/log40  &
root -l -b -q 'ana_ttbar.C("'$path_prefix'/ttcc-/Events/run_01_decayed_1/tag_1_delphes_events.root", "'$target_path'/ttcc-.root")'  &> ./$target_path/log/log41  &

root -l -b -q 'ana_ttbar.C("'$path_prefix'/ttjj+/Events/run_01_decayed_1/tag_1_delphes_events.root", "'$target_path'/ttjj+.root")'  &> ./$target_path/log/log60  &
root -l -b -q 'ana_ttbar.C("'$path_prefix'/ttjj-/Events/run_01_decayed_1/tag_1_delphes_events.root", "'$target_path'/ttjj-.root")'  &> ./$target_path/log/log61  &
