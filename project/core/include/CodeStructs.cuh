//
// Created by carlo on 2026-01-22.
//

#ifndef CODESTRUCTS_CUH
#define CODESTRUCTS_CUH

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <iomanip>
#include <sstream>
#include <string>

namespace CodeCuda
{

    constexpr int32_t k_matrix_pretty_print_limit = 128;
    constexpr int32_t k_matrix_summary_edge_count = 1;

    inline std::string FormatMatrixValue(float value)
    {
        std::ostringstream stream;
        stream << std::fixed << std::setprecision(3) << value;
        std::string formatted = stream.str();

        if (formatted.find('.') != std::string::npos)
        {
            while (!formatted.empty() && formatted.back() == '0')
            {
                formatted.pop_back();
            }
            if (!formatted.empty() && formatted.back() == '.')
            {
                formatted.pop_back();
            }
        }

        if (formatted == "-0")
        {
            return "0";
        }

        return formatted;
    }

    inline std::string FormatMatrixOutput(int32_t M, int32_t N, const float *data, bool force_full_output = false)
    {
        std::ostringstream output;
        output << "Shape: " << M << " x " << N << '\n';

        if (data == nullptr || M <= 0 || N <= 0)
        {
            output << "[]\n\n";
            return output.str();
        }

        const bool summarize_output = !force_full_output && (M * N) > k_matrix_pretty_print_limit;

        int32_t cell_width = 0;
        for (int32_t i = 0; i < M * N; ++i)
        {
            cell_width = std::max<int32_t>(cell_width, static_cast<int32_t>(FormatMatrixValue(data[i]).size()));
        }

        cell_width = std::max<int32_t>(cell_width, 2); // for "..."

        const auto should_print_row = [M, summarize_output](int32_t row)
        {
            if (!summarize_output || M <= k_matrix_summary_edge_count * 2)
                return true;

            return row < k_matrix_summary_edge_count || row >= M - k_matrix_summary_edge_count;
        };

        const auto should_print_col = [N, summarize_output](int32_t col)
        {
            if (!summarize_output || N <= k_matrix_summary_edge_count * 2)
                return true;

            return col < k_matrix_summary_edge_count || col >= N - k_matrix_summary_edge_count;
        };

        output << "[\n";

        bool printed_row_gap = false;

        for (int32_t row = 0; row < M; ++row)
        {
            if (!should_print_row(row))
            {
                if (!printed_row_gap)
                {
                    output << "  ...\n";
                    printed_row_gap = true;
                }
                continue;
            }

            printed_row_gap = false;

            output << "  [ ";

            bool first_cell = true;
            bool printed_col_gap = false;

            for (int32_t col = 0; col < N; ++col)
            {
                if (!should_print_col(col))
                {
                    if (!printed_col_gap)
                    {
                        if (!first_cell)
                            output << "  ";

                        output << std::setw(cell_width) << "...";

                        first_cell = false;
                        printed_col_gap = true;
                    }

                    continue;
                }

                if (!first_cell)
                    output << "  ";

                output << std::setw(cell_width) << FormatMatrixValue(data[row * N + col]);

                first_cell = false;
            }

            output << " ]\n";
        }

        output << "]\n\n";
        return output.str();
    }
    inline std::string FormatMatrixOutputNoPadding(int32_t M, int32_t N, const float *data,
                                                   bool force_full_output = false)
    {
        std::ostringstream output;
        output << "Shape: " << M << " x " << N << '\n';

        if (data == nullptr || M <= 0 || N <= 0)
        {
            output << "[]\n\n";
            return output.str();
        }

        const bool summarize_output = !force_full_output && (M * N) > k_matrix_pretty_print_limit;

        int32_t cell_width = 0;
        for (int32_t i = 0; i < M * N; ++i)
        {
            cell_width = std::max<int32_t>(cell_width, static_cast<int32_t>(FormatMatrixValue(data[i]).size()));
        }

        const auto should_print_row = [M, summarize_output](int32_t row)
        {
            if (!summarize_output || M <= (k_matrix_summary_edge_count * 2))
            {
                return true;
            }
            return row < k_matrix_summary_edge_count || row >= (M - k_matrix_summary_edge_count);
        };

        const auto should_print_col = [N, summarize_output](int32_t col)
        {
            if (!summarize_output || N <= (k_matrix_summary_edge_count * 2))
            {
                return true;
            }
            return col < k_matrix_summary_edge_count || col >= (N - k_matrix_summary_edge_count);
        };

        output << "[\n";
        bool skipped_rows = false;
        for (int32_t row = 0; row < M; ++row)
        {
            if (!should_print_row(row))
            {
                if (!skipped_rows)
                {
                    output << "  ...\n";
                    skipped_rows = true;
                }
                continue;
            }

            skipped_rows = false;
            output << "  [ ";
            bool skipped_cols = false;
            for (int32_t col = 0; col < N; ++col)
            {
                if (!should_print_col(col))
                {
                    if (!skipped_cols)
                    {
                        if (col > 0)
                        {
                            output << "  ";
                        }
                        output << "...";
                        skipped_cols = true;
                    }
                    continue;
                }

                if (col > 0)
                {
                    output << " ";
                }

                if (skipped_cols)
                {
                    skipped_cols = false;
                }

                output << FormatMatrixValue(data[row * N + col]);
            }
            output << " ]\n";
        }
        output << "]\n\n";
        return output.str();
    }
    struct c_matrix
    {
        c_matrix(const c_matrix &) = delete;
        c_matrix &operator=(const c_matrix &) = delete;
        c_matrix(int32_t M, int32_t N)
        {
            this->M = M;
            this->N = N;
            this->data_size = this->M * this->N;
            this->data = new float[M * N];
            this->data_t = new float[M * N];
            this->shape = new int[2];
            this->shape[0] = M;
            this->shape[1] = N;
        }
        c_matrix &Full(float val)
        {
            for (int32_t i = 0; i < this->data_size; i++)
            {
                data[i] = val;
            }
            return *this;
        }

        c_matrix &Rand()
        {
            srand(40);
            for (int32_t i = 0; i < this->data_size; i++)
            {
                data[i] = rand() % 1000;
            }
            return *this;
        }

        c_matrix &Full_Arange()
        {
            for (int32_t i = 0; i < this->data_size; i++)
            {
                data[i] = float(i);
            }
            return *this;
        }

        c_matrix &RandInt(int min, int max)
        {
            srand(40);
            for (int32_t i = 0; i < this->data_size; i++)
            {
                data[i] = float(min + (rand() % (max - min)));
            }
            return *this;
        }
        c_matrix &Rand(int min, int max)
        {
            srand(40);
            for (int32_t i = 0; i < this->data_size; i++)
            {
                float decimal = float(float(rand()) / float(RAND_MAX));
                data[i] = float(min + (rand() % (max - min))) + decimal;
            }
            return *this;
        }

        c_matrix &Arange_Sequential()
        {
            for (int32_t i = 0; i < this->data_size; i++)
            {
                data[i] = i;
            }
            return *this;
        }
        c_matrix &BuildTranpose()
        {
            for (int32_t i = 0; i < this->data_size; i++)
            {
                int32_t x = int32_t(i % N);
                int32_t y = int32_t(i / N);
                data_t[x * this->M + y] = data[i];
            }
            return *this;
        }
        c_matrix &Print(bool force_full_output = false, bool padded = true)
        {
            const std::string output = (padded) ? FormatMatrixOutput(M, N, data, force_full_output)
                                                : FormatMatrixOutputNoPadding(M, N, data, force_full_output);
            printf("%s", output.c_str());
            return *this;
        }
        ~c_matrix()
        {
            delete[] shape;
            delete[] data_t;
            delete[] data;
        }
        float *Get_Data() const
        {
            assert(data != nullptr);
            return data;
        }
        float *Get_T()
        {
            assert(data_t != nullptr);
            BuildTranpose();
            return data_t;
        }

        int32_t *Shape() { return shape; }
        static void PrintAnyMatrix(int32_t M, int32_t N, float *data, bool force_full_output = false,
                                   bool padded = true)
        {
            const std::string output = padded ? FormatMatrixOutput(M, N, data, force_full_output)
                                              : FormatMatrixOutputNoPadding(M, N, data, force_full_output);

            printf("%s", output.c_str());
        }

    private:
        float *data = nullptr;
        float *data_t = nullptr;
        int32_t M = 0;
        int32_t N = 0;
        int32_t *shape = nullptr;
        int32_t data_size = 0;
    };
    struct c_edge
    {
        float vec = 0.0f;
        bool is_wall = false;
        int GetState() { return is_wall ? 0 : 1; }
    };
    struct c_cell
    {
        float div = 0.0f;
        bool is_wall = false;
        int s = 0;
        float density = 1.0f;
        float pressure = 0.0f;
    };
    struct c_grid
    {
        c_grid(int width, int height)
        {

            this->edge_w = (width + 1);
            this->edge_h = (height + 1);
            this->w = width;
            this->h = height;
            this->dx = 1.0f / float(w);
            grid.reserve(width * height);
            u_edges.reserve(edge_w * edge_h);
            v_edges.reserve(edge_w * edge_h);

            for (int i = 0; i < u_edges.capacity(); ++i)
            {
                u_edges.emplace_back(c_edge{});
                int x = i % edge_w;
                int y = i / edge_w;
                u_edges[i].is_wall = x == 0 || x == edge_w - 1 || y == 0 || y == edge_h - 1;
            }
            for (int i = 0; i < v_edges.capacity(); ++i)
            {
                v_edges.emplace_back(c_edge{});
                int x = i % edge_w;
                int y = i / edge_w;
                v_edges[i].is_wall = x == 0 || x == edge_w - 1 || y == 0 || y == edge_h - 1;
                // int v = i % 2 == 0 ? -1 : 1;
                v_edges[i].vec = float(rand() % 1000) / 1000.0f;
            }

            for (int i = 0; i < grid.capacity(); ++i)
            {
                grid.emplace_back(c_cell{});
                int x = i % w;
                int y = i / w;
                grid[i].is_wall = x == 0 || x == w - 1 || y == 0 || y == h - 1;
                if (!grid[i].is_wall)
                {
                    valid_cell_count++;
                }
            }
            for (int i = 0; i <grid.size(); ++i)
            {
                c_edge *edge_u_left_out = nullptr;
                c_edge *edge_u_right_out = nullptr;
                c_edge *edge_v_top_out = nullptr;
                c_edge *edge_v_bottom_out = nullptr;
                int x = i % w;
                int y = i / w;
                GetCellEdges(x, y, edge_u_left_out, edge_u_right_out, edge_v_top_out, edge_v_bottom_out);
                grid[i].s = edge_u_left_out->GetState() + edge_u_right_out->GetState() + edge_v_top_out->GetState() + edge_v_bottom_out->GetState(); 
            }
        }
        void RunSimulation(int steps)
        {
            solved_grid_states.resize(steps);
            solved_grid_u.resize(steps);
            solved_grid_v.resize(steps);
            for (int i = 0; i < solved_grid_states.size(); ++i)
            {
                solved_grid_states[i].reserve(grid.size());
            }
            for (int i = 0; i < solved_grid_u.size(); ++i)
            {
                solved_grid_u[i].reserve(u_edges.size());
            }
            for (int i = 0; i < solved_grid_v.size(); ++i)
            {
                solved_grid_v[i].reserve(v_edges.size());
            }
            
            for (int i = 0; i < steps; ++i)
            {
                Update();
                memcpy(solved_grid_states[i].data(), grid.data(), grid.size() * sizeof(c_cell));
                memcpy(solved_grid_u[i].data(), u_edges.data(), u_edges.size() * sizeof(c_edge));
                memcpy(solved_grid_v[i].data(), v_edges.data(), v_edges.size() * sizeof(c_edge));
            }
            
        }
        void Update()
        {
            UpdateStep();
            UpdateDiv();
        }
        void UpdateStep()
        {
            for (int i = 0; i < v_edges.size(); ++i)
            {
                if (v_edges[i].is_wall)
                {
                    continue;
                }
                v_edges[i].vec += (t * g);
            }
        }
        void UpdateDiv()
        {
            int converged = 0;
            int idx = 0;
            for (int i = 0; i < grid.size(); ++i)
            {
                if (grid[i].is_wall)
                {
                    continue;
                }
                grid[i].pressure = 0.0f;
            }
            while (converged < valid_cell_count)
            {
                    for (int i = 0; i < grid.size(); ++i)
                {
                    if (grid[i].is_wall)
                    {
                        continue;
                    }
                    c_edge *edge_u_left_out = nullptr;
                    c_edge *edge_u_right_out = nullptr;
                    c_edge *edge_v_top_out = nullptr;
                    c_edge *edge_v_bottom_out = nullptr;
                    int x = i % w;
                    int y = i / w;
                    GetCellEdges(x, y, edge_u_left_out, edge_u_right_out, edge_v_top_out, edge_v_bottom_out);
                    grid[i].div =
                        Overrelaxation(edge_u_right_out->vec - edge_u_left_out->vec + edge_v_top_out->vec - edge_v_bottom_out->vec);
                    int s = grid[i].s;
                    grid[i].pressure += (grid[i].div/s) * (grid[i].density * dx / t);
                    if (s == 0)
                    {
                        CODECUDA_PRINTLN("State must be at least one");
                        continue;
                    }
                    edge_u_left_out->vec += grid[i].div *  edge_u_left_out->GetState() / s;
                    edge_u_right_out->vec -= grid[i].div *  edge_u_right_out->GetState() / s;
                    edge_v_bottom_out->vec += grid[i].div *  edge_v_bottom_out->GetState() / s;
                    edge_v_top_out->vec -= grid[i].div *  edge_v_top_out->GetState() / s;
                }
                
                converged = 0;
                for (int i = 0; i < grid.size(); ++i)
                {
                    if (grid[i].is_wall)
                    {
                        continue;
                    }
                    if (std::abs(grid[i].div) < epsilon)
                    {
                        converged++;
                    };
                }
                idx++;
            }
            PrintDivergenceConvergence(idx);
        }


        void GetCellEdges(int x, int y, c_edge *& edge_u_left_out, c_edge *& edge_u_right_out, c_edge *&edge_v_top_out,
                          c_edge *&edge_v_bottom_out)
        {

            edge_u_left_out = &u_edges[y * edge_w + x];
            edge_u_right_out = &u_edges[y * edge_w + (x + 1)];

            edge_v_top_out = &v_edges[y * edge_w + x];
            edge_v_bottom_out = &v_edges[(y + 1) * edge_w + x];
        }
        float Overrelaxation(float div)
        {
            return div * 1.9f;
        }
        void PrintDivergenceConvergence(int iteration)
        {
            int total = 0;
            int converged = 0;
            float maxAbsDiv = 0.0f;
            float sumAbsDiv = 0.0f;

            for (int i = 0; i < grid.size(); ++i)
            {
                if (grid[i].is_wall)
                    continue;

                int x = i % w;
                int y = i / w;

                float div = grid[i].div;
                float absDiv = std::abs(div);

                total++;
                sumAbsDiv += absDiv;
                maxAbsDiv = max(maxAbsDiv, absDiv);

                if (absDiv < epsilon)
                    converged++;
            }

            float avgAbsDiv = total > 0 ? sumAbsDiv / float(valid_cell_count) : 0.0f;

            std::cout
                << "iter " << iteration
                << " | converged " << converged << "/" << total
                << " | avgDiv " << avgAbsDiv
                << " | maxDiv " << maxAbsDiv
                << "\n";
        }
        const float epsilon = 0.0001f;
        float t = 1.0f / 30.0f;
        float g = -9.81f;
        int w = -1;
        int h = -1;
        int valid_cell_count = 0;
        int edge_w = -1;
        int edge_h = -1;
        float dx = 0.0f;
        float dy = 0.0f;
        std::vector<c_cell> grid;
        std::vector<c_edge> u_edges;
        std::vector<c_edge> v_edges;
        std::vector<std::vector<c_cell>> solved_grid_states;
        std::vector<std::vector<c_cell>> solved_grid_u;
        std::vector<std::vector<c_cell>> solved_grid_v;
    };
    


} // namespace CodeCuda


#endif // CODESTRUCTS_CUH
