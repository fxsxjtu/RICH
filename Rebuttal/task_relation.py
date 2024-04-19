from scipy.stats import pearsonr
import numpy as np
datas = ['NY', 'CH']
for data in datas:
        od_matrix = np.load('../ny_data/final_data/od_matrix.npy')
        price_label = np.load('../ny_data/final_data/bnb_price.npy')[:99].flatten().tolist()
        bnb_label = np.load('../ny_data/final_data/bnb_label.npy')[:99].flatten().tolist()
        flow_labels = np.stack([np.sum(od_matrix, axis=1), np.sum(od_matrix, axis=2)]).transpose([1, 2, 0])
        in_flow = flow_labels[:, :, 0].flatten().tolist()
        out_flow = flow_labels[:, :, 1].flatten().tolist()

        corr_coefficient_price_in, _ = pearsonr(in_flow, price_label)
        corr_coefficient_bnb_in, _ = pearsonr(in_flow, bnb_label)
        corr_coefficient_price_out, _ = pearsonr(out_flow, price_label)
        corr_coefficient_bnb_out, _ = pearsonr(out_flow, bnb_label)
        corr_coefficient_tasks, _ = pearsonr(price_label, bnb_label)
        print('The PCC between Inflow and NYBnbPrice: ', corr_coefficient_price_in)
        print('The PCC between Inflow and NYBnbReservation: ', corr_coefficient_bnb_in)
        print('The PCC between Outflow and NYBnbPrice: ', corr_coefficient_price_out)
        print('The PCC between Outflow and NYBnbReservation: ', corr_coefficient_bnb_out)
        print('The PCC between NYBnbReservation and NYBnbPrice: ', corr_coefficient_tasks)

        od_matrix = np.load('../ch_data/final_data/od_matrix.npy')
        price_label = np.load('../ch_data/final_data/bnb_price.npy')[:99].flatten().tolist()
        bnb_label = np.load('../ch_data/final_data/bnb_label.npy')[:99].flatten().tolist()
        flow_labels = np.stack([np.sum(od_matrix, axis=1), np.sum(od_matrix, axis=2)]).transpose([1, 2, 0])
        in_flow = flow_labels[:, :, 0].flatten().tolist()
        out_flow = flow_labels[:, :, 1].flatten().tolist()

        corr_coefficient_price_in, _ = pearsonr(in_flow, price_label)
        corr_coefficient_bnb_in, _ = pearsonr(in_flow, bnb_label)
        corr_coefficient_price_out, _ = pearsonr(out_flow, price_label)
        corr_coefficient_bnb_out, _ = pearsonr(out_flow, bnb_label)
        corr_coefficient_tasks, _ = pearsonr(price_label, bnb_label)
        print('The PCC between Inflow and CHBnbPrice: ', corr_coefficient_price_in)
        print('The PCC between Inflow and CHBnbReservation: ', corr_coefficient_bnb_in)
        print('The PCC between Outflow and CHBnbPrice: ', corr_coefficient_price_out)
        print('The PCC between Outflow and CHBnbReservation: ', corr_coefficient_bnb_out)
        print('The PCC between CHBnbReservation and CHBnbPrice: ', corr_coefficient_tasks)




