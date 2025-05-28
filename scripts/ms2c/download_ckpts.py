from fragnnet.utils.misc_utils import get_best_ckpt_from_wandb
	
if __name__ == "__main__":

	save_dp = "./saved_ckpts/"

	fragnnet_runs = [
		'yylzfd6t', # nist20v3_inchikey_fraggnn_d3_s0
		'ucun3t92', # nist20v3_inchikey_fraggnn_d3_s1
		'mm6lzxf3', # nist20v3_inchikey_fraggnn_d3_s2 
		'plpayglr', # nist20v3_inchikey_fraggnn_d3_s3
		'24mvykw7', # nist20v3_inchikey_fraggnn_d3_s4
		
		'a64wr0sb', # nist20v3_inchikey_fraggnn_d4_s0
		'zad13roi', # nist20v3_inchikey_fraggnn_d4_s1
		'xqbltkvt', # nist20v3_inchikey_fraggnn_d4_s2 
		'9slcb73m', # nist20v3_inchikey_fraggnn_d4_s3
		'uqznj3a7', # nist20v3_inchikey_fraggnn_d4_s4
		
		'ad8s7pkj', # nist20v3_scaffold_fraggnn_d3_s0
		'qcmvlegm', # nist20v3_scaffold_fraggnn_d3_s1
		'n9nignyj', # nist20v3_scaffold_fraggnn_d3_s2 
		'ty6i1ji0', # nist20v3_scaffold_fraggnn_d3_s3
		'brn49wh7', # nist20v3_scaffold_fraggnn_d3_s4
		
		'wfm2tkrx', # nist20v3_inchikey_fraggnn_d3_s0
		'hwan9uas', # nist20v3_inchikey_fraggnn_d3_s1
		'1hfhywj3', # nist20v3_inchikey_fraggnn_d3_s2 
		'zwxbopwo', # nist20v3_inchikey_fraggnn_d3_s3
		'inmmhv8v', # nist20v3_inchikey_fraggnn_d3_s4
	]
	
	for run_id in fragnnet_runs:
		get_best_ckpt_from_wandb(save_dp, run_id, use_cached=True)


	iceberg_runs = [
		
		# nist20v3_inchikey_iceberg_gen
		'i7k9gc3d', # nist20v3_inchikey_iceberg_gen_s0
		'mz3s3h62', # nist20v3_inchikey_iceberg_gen_s1
		'6f4nes2g', # nist20v3_inchikey_iceberg_gen_s2
		'lp60e1nq', # nist20v3_inchikey_iceberg_gen_s3
		'3zjhzrss', # nist20v3_inchikey_iceberg_gen_s4

		# nist20v3_inchikey_iceberg_inten_100  
		'c8iwu7il', # nist20v3_inchikey_iceberg_inten_100_s0
		'rtt8i90g', # nist20v3_inchikey_iceberg_inten_100_s1
		'acgyna2a', # nist20v3_inchikey_iceberg_inten_100_s2
		'683gijym', # nist20v3_inchikey_iceberg_inten_100_s3
		'yeatgz7o', # nist20v3_inchikey_iceberg_inten_100_s4
	
		# nist20v3_scaffold_iceberg_gen
		'4pweeg45', # nist20v3_scaffold_iceberg_gen_s0
		'9j26mr4s', # nist20v3_scaffold_iceberg_gen_s1
		'n27tube2', # nist20v3_scaffold_iceberg_gen_s2
		'biofdpxl', # nist20v3_scaffold_iceberg_gen_s3
		'cyot56r3', # nist20v3_scaffold_iceberg_gen_s4
		
		# nist20v3_scaffold_iceberg_inten_100
		'riyr41po', # nist20v3_scaffold_iceberg_inten_100_s0
		'pmaom333', # nist20v3_scaffold_iceberg_inten_100_s4 
		's37eax0j', # nist20v3_scaffold_iceberg_inten_100_s1
		'x2l4adze', # nist20v3_scaffold_iceberg_inten_100_s2
		'6w68jgf6', # nist20v3_scaffold_iceberg_inten_100_s3
		
	]

	for run_id in iceberg_runs:
		get_best_ckpt_from_wandb(save_dp, run_id, use_cached=True)
  

	massformer_runs = [
		'2ay7zwbm', # nist20v3_inchikey_massformer_s0
		'huab7u4w', # nist20v3_inchikey_massformer_s1
		'3w8sxkx2', # nist20v3_inchikey_massformer_s2 
		'x6xev9hu', # nist20v3_inchikey_massformer_s3
		'24mvykw7', # nist20v3_inchikey_massformer_s4
		
		'wf1zkb7n', # nist20v3_inchikey_massformer_s0
		'p06j6ijs', # nist20v3_inchikey_massformer_s1
		'fgbtd9us', # nist20v3_inchikey_massformer_s2 
		'6x3ar9d5', # nist20v3_inchikey_massformer_s3
		'qtqkl08n', # nist20v3_inchikey_massformer_s4
	]
	
	for run_id in massformer_runs:
		get_best_ckpt_from_wandb(save_dp, run_id)


	nemis_runs = [
		'ylm3miyr', # nist20v3_inchikey_neims_s0
		'ebhx6zkd', # nist20v3_inchikey_neims_s1
		'r6ks7h3r', # nist20v3_inchikey_neims_s2
		'spy7ce14', # nist20v3_inchikey_neims_s3
		'e1e479z2', # nist20v3_inchikey_neims_s4

		'n28n8vzx', # nist20v3_scaffold_neims_s0
		'53gqo63a', # nist20v3_scaffold_neims_s1
		'qqgti9my', # nist20v3_scaffold_neims_s2
		'uft1aqrp', # nist20v3_scaffold_neims_s3
		'ik4u88ol', # nist20v3_scaffold_neims_s4
	]

	for run_id in nemis_runs:
		get_best_ckpt_from_wandb(save_dp, run_id)