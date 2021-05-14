defmodule LinearRegression do
  import Nx.Defn

  #function calculates residual sum of squares
  defn rss({a, b}, x, y) do
    y_pred = a * x + b
    Nx.mean(Nx.power(y - y_pred, 2))
  end

  #function updates
  defn update({a, b} = params, input, target, lr) do
    {grad_m, grad_b} = grad(params, &rss(&1, input, target))
    {a - grad_m * lr, b - grad_b * lr}
  end

  #evaluate training process
  def train(epochs, data, lr, batch_size) do
    init_params = {Nx.random_normal({}, 0.0, 0.1), Nx.random_normal({}, 0.0, 0.1)}
    for _ <- 1..epochs, reduce: init_params do
      acc ->
        data
        |> Enum.take(batch_size)
        |> Enum.reduce(
             acc,
             fn batch, curr_params ->
               {input, target} = Enum.unzip(batch)
               x = Nx.tensor(input)
               y = Nx.tensor(target)
               update(curr_params, x, y, lr)
             end
           )
    end
  end
end

target_a = :rand.normal(0.0, 10.0)
target_b = :rand.normal(0.0, 5.0)
target_fn = fn x -> target_a * x + target_b end
data =
  Stream.repeatedly(fn -> for _ <- 1..32, do: :rand.uniform() * 10 end)
  |> Stream.map(fn x -> Enum.zip(x, Enum.map(x, target_fn)) end)
IO.puts("Target a: #{target_a}\tTarget b: #{target_b}\n")
{a, b} = LinReg.train(100, data, 0.01, 200)
IO.puts("Learned a: #{Nx.to_scalar(a)}\tLearned b: #{Nx.to_scalar(b)}")


defmodule Fun do
  import Nx.Defn

  defn logistic(x, beta) do
    Nx.divide(1, Nx.add(1, Nx.exp(Nx.multiply(x, -beta))))
  end

  def logistic1(x, beta) do
    Nx.divide(1, Nx.add(1, Nx.exp(Nx.multiply(x, -beta))))
  end

  defn softmax(t) do
    Nx.exp(t) / Nx.sum(Nx.exp(t))
  end

  @defn_compiler {EXLA, client: :host}
  defn softmax(t) do
    Nx.exp(t) / Nx.sum(Nx.exp(t))
  end

  def softmax1(t) do
    Nx.exp(t) / Nx.sum(Nx.exp(t))
  end

  def determinant(matrix) do
    {size, _} = Nx.shape(matrix)
    mat_factorized = Nx.LinAlg.lu(matrix) |> elem(2)
    diag = Nx.tensor(for i <- 0..(size-1) do Nx.to_scalar(mat_factorized[i][i]) end)
    Nx.multiply(-1, Nx.reduce(diag, 1, fn x, acc -> Nx.multiply(x, acc) end))
  end

end
