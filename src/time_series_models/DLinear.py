import torch
import torch.nn as nn
import torch.nn.functional as F
from src.time_series_models.layers.Autoformer_EncDec import series_decomp


class Model(nn.Module):
    """
    DLinear with exogenous fusion from x_mark_dec (a.k.a. batch_y_mark)
    Paper: https://arxiv.org/pdf/2205.13504.pdf
    """

    def __init__(self, configs, individual=False):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = (
            configs.seq_len
            if self.task_name in ['classification', 'anomaly_detection', 'imputation']
            else configs.pred_len
        )
        self.decompsition = series_decomp(configs.moving_avg)
        self.individual = individual
        self.channels = configs.enc_in

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Seasonal[i].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len])
                )
                self.Linear_Trend[i].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len])
                )
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Seasonal.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len])
            )
            self.Linear_Trend.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len])
            )

        # Always try to use exogenous fusion if x_mark_dec is provided
        self.use_exog = True

        # Initialize exog_fusion here using configs.exog_dim
        # print('TRUE 1', configs)
        if self.use_exog and hasattr(configs, 'exog_dim'):
            # print(configs.exog_dim, self.channels)
            self.exog_fusion = nn.Sequential(
                nn.Linear(configs.exog_dim, self.channels),
                nn.ReLU(),
                nn.Linear(self.channels, self.channels)
            )
            self.exog_gate = nn.Sequential(
                nn.Linear(configs.exog_dim, self.channels),
                nn.Sigmoid()
            )
        else:
            self.exog_fusion = None  # fallback, no exogenous fusion

        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.enc_in * configs.seq_len, configs.num_class)

    def encoder(self, x):
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)

        if self.individual:
            seasonal_output = torch.zeros(
                [seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                dtype=seasonal_init.dtype,
                device=seasonal_init.device,
            )
            trend_output = torch.zeros(
                [trend_init.size(0), trend_init.size(1), self.pred_len],
                dtype=trend_init.dtype,
                device=trend_init.device,
            )
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        x = seasonal_output + trend_output
        return x.permute(0, 2, 1)  # [B, pred_len, D]

    def forecast(self, x_enc, x_mark_dec=None):
        y_hat = self.encoder(x_enc)  # [B, pred_len, D]
        # print(self.use_exog, x_mark_dec.shape, self.exog_fusion)
        # print('TRUE', self.use_exog and x_mark_dec is not None and self.exog_fusion is not None)
        if self.use_exog and x_mark_dec is not None and self.exog_fusion is not None:
            exog_adj = self.exog_fusion(x_mark_dec)
            gate = self.exog_gate(x_mark_dec)
            # gate = 0
            y_hat = y_hat * (1 - gate) + exog_adj * gate


        return y_hat

    def imputation(self, x_enc):
        return self.encoder(x_enc)

    def anomaly_detection(self, x_enc):
        return self.encoder(x_enc)

    def classification(self, x_enc):
        enc_out = self.encoder(x_enc)
        output = enc_out.reshape(enc_out.shape[0], -1)
        return self.projection(output)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            # print('x_mark_dec', x_mark_dec.shape)
            dec_out = self.forecast(x_enc, x_mark_dec)
            # print('dec_out', dec_out)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        elif self.task_name == 'imputation':
            return self.imputation(x_enc)
        elif self.task_name == 'anomaly_detection':
            return self.anomaly_detection(x_enc)
        elif self.task_name == 'classification':
            return self.classification(x_enc)
        return None
