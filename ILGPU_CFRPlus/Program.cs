
namespace ILGPU_CFRPlus
{
    using ILGPU;
    using ILGPU.Algorithms;
    using ILGPU.Algorithms.Random;
    using ILGPU.Runtime;
    using System;
    using System.Runtime.InteropServices;

    public static class Program
    {

        // Used in ILGPU
        static int size = 100; //10000; //26700; 
        static Random rng = new Random();

        #region "Original CFR+"

        static bool discount = true;
        static int delay = 0; //100;
        static int wmode = 2;
        static int iterationCount = 0;
        static double[] payoffs;
        static double[][] strategy;
        static double[][] cfr;

        static void MatrixGame()
        {

            payoffs = new double[size * size];

            strategy = new double[2][];
            cfr = new double[2][];

            for (int player = 0; player < 2; player++)
            {
                strategy[player] = new double[size];
                cfr[player] = new double[size];
            }

            // Just a uniform distribution
            for (int a = 0; a < size; a++)
                for (int b = 0; b < size; b++)
                    payoffs[a * size + b] = rng.NextDouble() * 2 - 1;

        }

        static double GetPayoff(int player, int a, int b)
        {
            if (player == 0)
                return payoffs[a * size + b];
            else
                return -payoffs[b * size + a];
        }

        static double[] GetCurrentStrategy(int player)
        {

            double[] cs = new double[size];
            double sum = 0;

            for (int i = 0; i < size; i++)
                sum += Math.Max(0.0, cfr[player][i]);

            if (sum > 0)
            {
                for (int i = 0; i < size; i++)
                    cs[i] = cfr[player][i] > 0 ? cfr[player][i] / sum : 0.0;
            }
            else
            {
                for (int i = 0; i < size; i++)
                    cs[i] = 1.0 / size;
            }

            return cs;
        }

        static void CFR(int player, double alpha, double beta)
        {
            double[] cfu = new double[size];
            double ev = 0;

            double[] sp = GetCurrentStrategy(player);
            double[] so = GetCurrentStrategy(player ^ 1);

            for (int a = 0; a < size; a++)
            {
                cfu[a] = 0;

                for (int b = 0; b < size; b++)
                {
                    cfu[a] += so[b] * GetPayoff(player, a, b);
                }

                ev += sp[a] * cfu[a];
            }

            if (discount)
            {
                double coef;

                for (int a = 0; a < size; a++)
                {
                    cfr[player][a] += cfu[a] - ev;

                    if (cfr[player][a] >= 0)
                        coef = Math.Pow(iterationCount, alpha);
                    else
                        coef = Math.Pow(iterationCount, beta);

                    cfr[player][a] *= coef / (coef + 1);

                }

            }
            else
            {
                for (int a = 0; a < size; a++)
                    cfr[player][a] = Math.Max(0.0, cfr[player][a] + cfu[a] - ev);
            }


            double w;

            if (iterationCount > delay)
            {
                int t = iterationCount - delay;

                switch (wmode)
                {
                    case 0: w = 1; break;
                    case 1: w = t; break;
                    default: w = t * t; break;
                }
            }
            else
            {
                w = 0;
            }

            for (int a = 0; a < size; a++)
                strategy[player][a] += sp[a] * w;
        }

        #endregion 


        // ======================== ILGPU ========================

        //Kernel_Sums(size, sums.View, p1_strategy.View, p2_strategy.View);
        static void Kernel_Sum(Index1D z, ArrayView<double> sums, ArrayView<double> p1, ArrayView<double> p2)
        {
            Atomic.Add(ref sums[0], p1[z]);
            Atomic.Add(ref sums[1], p2[z]);
        }

        //Kernel_Positive_Sums(size, sums.View, p1_cfr.View, p2_cfr.View);
        static void Kernel_Positive_Sum(Index1D z, ArrayView<double> sums, ArrayView<double> p1, ArrayView<double> p2)
        {
            if (p1[z] > 0) Atomic.Add(ref sums[0], p1[z]);
            if (p2[z] > 0) Atomic.Add(ref sums[1], p2[z]);
        }

        //Kernel_Avg_Strats(size, p1_avg.View, p2_avg.View, p1_cfr.View, p2_cfr.View, sums.View);
        static void Kernel_Get_Avg_Strats(Index1D z, ArrayView<double> p1_avg, ArrayView<double> p2_avg,
            ArrayView<double> p1_cfr, ArrayView<double> p2_cfr, ArrayView<double> sums)
        {
            if (sums[0] > 0)
            {
                if (p1_cfr[z] > 0) p1_avg[z] = p1_cfr[z] / sums[0]; else p1_avg[z] = 0.0;
            }
            else
            {
                p1_avg[z] = 1.0 / p1_avg.Length;
            }

            if (sums[1] > 0)
            {
                if (p2_cfr[z] > 0) p2_avg[z] = p2_cfr[z] / sums[1]; else p2_avg[z] = 0.0;
            }
            else
            {
                p2_avg[z] = 1.0 / p2_avg.Length;
            }

        }

        //Kernel_Norm_Strats(size, p1_norm.View, p2_norm.View, p1_strategy.View, p2_strategy.View, sums.View);
        static void Kernel_Get_Norm_Strats(Index1D z, ArrayView<double> p1_norm, ArrayView<double> p2_norm,
            ArrayView<double> p1_strat, ArrayView<double> p2_strat, ArrayView<double> sums)
        {
            if (sums[0] > 0)
                p1_norm[z] = p1_strat[z] / sums[0];
            else
                p1_norm[z] = 1.0 / p1_norm.Length;

            if (sums[1] > 0)
                p2_norm[z] =  p2_strat[z] / sums[1];
            else
                p2_norm[z] = 1.0 / p2_norm.Length;
        }

        //if (player == 0)
        //    return payoffs[a * size + b];
        //else
        //    return -payoffs[b * size + a];
        //ev.MemSetToZero();
        //Kernel_CFU_P1(size, cfu.View, ev.View, p1_avg.View, p2_avg.View, mpayoffs.View);
        static void Kernel_Get_CFU_P1(Index1D z, ArrayView<double> cfu, ArrayView<double> ev,
            ArrayView<double> p1_avg,  ArrayView<double> p2_avg, ArrayView<double> payoffs)
        {
            cfu[z] = 0.0;
            for (int b = 0; b < p2_avg.Length; b++)
                cfu[z] += p2_avg[b] * payoffs[z * p2_avg.Length + b];
            //ev[0] += p1_avg[z] * cfu[z];
            Atomic.Add(ref ev[0], p1_avg[z] * cfu[z]);
        }

        //ev.MemSetToZero();
        //Kernel_CFU_P2(size, cfu.View, ev.View, p1_avg.View, p2_avg.View, mpayoffs.View);
        static void Kernel_Get_CFU_P2(Index1D z, ArrayView<double> cfu, ArrayView<double> ev,
            ArrayView<double> p1_avg, ArrayView<double> p2_avg, ArrayView<double> payoffs)
        {
            cfu[z] = 0.0;
            for (int b = 0; b < cfu.Length; b++)
                cfu[z] += p1_avg[b] * -payoffs[b * p1_avg.Length + z];
            //ev[0] += p2_avg[z] * cfu[z];
            Atomic.Add(ref ev[0], p2_avg[z] * cfu[z]);
        }

        //Kernel_CFR(size, p1_cfr.View, cfu.View, ev.View, ic.View);
        static void Kernel_Get_CFR(Index1D z, ArrayView<double> p_cfr, ArrayView<double> cfu, ArrayView<double> ev, ArrayView<int> ic)
        {
            const bool discount = false;
            const double alpha = 3 / 2;
            const double beta = 0;

            double newcfr = p_cfr[z] + cfu[z] - ev[0];

            if (discount)
            {               
                if (newcfr > 0)
                {
                    double coef = XMath.Pow(ic[0], alpha);
                    p_cfr[z] = newcfr * (coef / (coef + 1));
                }
                else
                {
                    double coef = XMath.Pow(ic[0], beta);
                    p_cfr[z] = newcfr * (coef / (coef + 1));
                }
            } 
            else
            {
                if (newcfr > 0) p_cfr[z] = newcfr; else p_cfr[z] = 0.0;
            }

        }

        //Kernel_Strats(size, p1_strategy.View, p1_avg.View, ic.View);
        static void Kernel_Get_Strategy(Index1D z, ArrayView<double> p_strategy, ArrayView<double> p_avg, ArrayView<int> ic)
        {
            double w;
            const int delay = 0;// 100;
            const int wmode = 2;

            if (ic[0] > delay)
            {
                int t = ic[0] - delay;

                switch (wmode)
                {
                    case 0: w = 1; break;
                    case 1: w = t; break;
                    default: w = t * t; break;
                }
            }
            else
            {
                w = 0;
            }

            p_strategy[z] += p_avg[z] * w;
        }

        //Kernel_BRs(size, brs.View, p1_norm.View, p2_norm.View, mpayoffs.View);
        static void Kernel_Get_BRs(Index1D z, ArrayView<double> brs, ArrayView<double> p1_norm, ArrayView<double> p2_norm, 
            ArrayView<double> payoffs)
        {

            double sum = 0;
            for (int b = 0; b < p1_norm.Length; b++)
                sum += p2_norm[b] * payoffs[z * p1_norm.Length + b];
            //if (sum > brs[0]) brs[0] = sum;
            Atomic.Max(ref brs[0], sum);

            sum = 0;
            for (int b = 0; b < p2_norm.Length; b++)
                sum += p1_norm[b] * -payoffs[b * p2_norm.Length + z];
            //if (sum > brs[1]) brs[1] = sum;
            Atomic.Max(ref brs[1], sum);

        }

        //Kernel_Create_Payoffs(size * size, mpayoffs.View, xr.GetView(accelerator.WarpSize));
        static void Kernel_Fill_Payoffs(Index1D z, ArrayView<double> payoffs, RNGView<XorShift128> rnd)
        {
            payoffs[z] = rnd.NextDouble() * 2 - 1;
        }

        // Page locked memory can be transfered between the GPU much faster.
        // ILGPU also has page locked arrays: AllocatePageLocked1D<double>
        static void copyPageLockedPinned( ref MemoryBuffer1D<double, Stride1D.Dense> mb, ref double[] ar, ref Accelerator acc)
        {
            GCHandle handle = GCHandle.Alloc(ar, GCHandleType.Pinned);

            try
            {
                AcceleratorStream stream = acc.DefaultStream;
                PageLockScope<double> scope = acc.CreatePageLockFromPinned(ar);
                mb.View.CopyFromPageLockedAsync(stream, scope);
                stream.Synchronize();
            }
            catch (Exception e)
            {
                Console.WriteLine(e.Message);
                Console.ReadLine();
            }
            finally
            {
                handle.Free();
            }

        }

        static void Main()
        {

            // Initialize ILGPU.
            //Context context = Context.CreateDefault();

            Context context = Context.Create(builder => builder.AllAccelerators().EnableAlgorithms());
            //Context context = Context.Create(builder => builder.Default().EnableAlgorithms());

            foreach (Device device in context.Devices)
            {
                Console.WriteLine(device);
            }

            //Console.ReadLine();

            Accelerator accelerator = context.GetPreferredDevice(preferCPU: false)
                                      .CreateAccelerator(context);

            //Accelerator accelerator = context.CreateCudaAccelerator(0);

            //Accelerator accelerator = context.CreateCPUAccelerator(0);

            // Create variables used.

            MemoryBuffer1D<double, Stride1D.Dense> p1_cfr = accelerator.Allocate1D<double>(size);
            MemoryBuffer1D<double, Stride1D.Dense> p2_cfr = accelerator.Allocate1D<double>(size);
            MemoryBuffer1D<double, Stride1D.Dense> p1_strategy = accelerator.Allocate1D<double>(size);
            MemoryBuffer1D<double, Stride1D.Dense> p2_strategy = accelerator.Allocate1D<double>(size);
            
            MemoryBuffer1D<double, Stride1D.Dense> mpayoffs = accelerator.Allocate1D<double>(size * size);      

            MemoryBuffer1D<double, Stride1D.Dense> sums = accelerator.Allocate1D<double>(2);
            MemoryBuffer1D<double, Stride1D.Dense> brs = accelerator.Allocate1D<double>(2);
            MemoryBuffer1D<double, Stride1D.Dense> ev = accelerator.Allocate1D<double>(1);
            MemoryBuffer1D<double, Stride1D.Dense> p1_avg = accelerator.Allocate1D<double>(size);
            MemoryBuffer1D<double, Stride1D.Dense> p2_avg = accelerator.Allocate1D<double>(size);
            MemoryBuffer1D<double, Stride1D.Dense> p1_norm = accelerator.Allocate1D<double>(size);
            MemoryBuffer1D<double, Stride1D.Dense> p2_norm = accelerator.Allocate1D<double>(size);
            MemoryBuffer1D<double, Stride1D.Dense> cfu = accelerator.Allocate1D<double>(size);
            MemoryBuffer1D<int, Stride1D.Dense> ic = accelerator.Allocate1D<int>(1);
           
            // Create Kernels.
            Action<Index1D, ArrayView<double>, ArrayView<double>, ArrayView<double>> Kernel_Sums =
                accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<double>, ArrayView<double>, ArrayView<double>>(Kernel_Sum);

            Action<Index1D, ArrayView<double>, ArrayView<double>, ArrayView<double>> Kernel_Positive_Sums =
                accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<double>, ArrayView<double>, ArrayView<double>>(Kernel_Positive_Sum);

            Action<Index1D, ArrayView<double>, ArrayView<double>, ArrayView<double>, ArrayView<double>, ArrayView<double>> Kernel_Avg_Strats =
                accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<double>, ArrayView<double>, ArrayView<double>, ArrayView<double>, ArrayView<double>>(Kernel_Get_Avg_Strats);

            Action<Index1D, ArrayView<double>, ArrayView<double>, ArrayView<double>, ArrayView<double>, ArrayView<double>> Kernel_Norm_Strats =
                accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<double>, ArrayView<double>, ArrayView<double>, ArrayView<double>, ArrayView<double>>(Kernel_Get_Norm_Strats);

            Action<Index1D, ArrayView<double>, ArrayView<double>, ArrayView<double>, ArrayView<double>, ArrayView<double>> Kernel_CFU_P1 =
                accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<double>, ArrayView<double>, ArrayView<double>, ArrayView<double>, ArrayView<double>>(Kernel_Get_CFU_P1);

            Action<Index1D, ArrayView<double>, ArrayView<double>, ArrayView<double>, ArrayView<double>, ArrayView<double>> Kernel_CFU_P2 =
                accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<double>, ArrayView<double>, ArrayView<double>, ArrayView<double>, ArrayView<double>>(Kernel_Get_CFU_P2);

            Action<Index1D, ArrayView<double>, ArrayView<double>, ArrayView<double>, ArrayView<int>> Kernel_CFR =
                accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<double>, ArrayView<double>, ArrayView<double>, ArrayView<int>>(Kernel_Get_CFR);

            Action<Index1D, ArrayView<double>, ArrayView<double>, ArrayView<int>> Kernel_Strats =
                accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<double>, ArrayView<double>, ArrayView<int>>(Kernel_Get_Strategy);

            Action<Index1D, ArrayView<double>, ArrayView<double>, ArrayView<double>, ArrayView<double>> Kernel_BRs =
                accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<double>, ArrayView<double>, ArrayView<double>, ArrayView<double>>(Kernel_Get_BRs);

            Action<Index1D, ArrayView<double>, RNGView<XorShift128>> Kernel_Create_Payoffs =
                accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<double>, RNGView<XorShift128>>(Kernel_Fill_Payoffs);

            
            //// This only works with a GPU and it breaks down for larger arrays (e.g. of size 10000x10000),
            //// the output of NextDouble() starts becoming extremely small.
            //RNG<XorShift128> xr = RNG.Create<XorShift128>(accelerator, rng);
            //Kernel_Create_Payoffs(size * size, mpayoffs.View, xr.GetView(accelerator.WarpSize));
            //accelerator.Synchronize();
            //double[] po = mpayoffs.GetAsArray1D();
            //for (int i=0; i<100; i++)
            //    Console.Write(po[i] + ", ");
            //Console.ReadLine();
          
            double[] poffs = new double[size * size];
            for (int i = 0; i < poffs.Length; i++)
                poffs[i] = rng.NextDouble() * 2 -1;
            // Not a big deal here but it would be important if swapping memory frequently.
            copyPageLockedPinned(ref mpayoffs, ref poffs, ref accelerator);
            poffs = null;

            Console.WriteLine("Starting...");

            // Do some CFR+
            for (int i = 0; i < 1000; i++)
            {

                // Update the iteration count.
                ic.CopyFromCPU(new int[] { i }); accelerator.Synchronize();

                // Update player one.
                sums.MemSetToZero(); accelerator.Synchronize();
                Kernel_Positive_Sums(size, sums.View, p1_cfr.View, p2_cfr.View); accelerator.Synchronize();
                Kernel_Avg_Strats(size, p1_avg.View, p2_avg.View, p1_cfr.View, p2_cfr.View, sums.View); accelerator.Synchronize();
                ev.MemSetToZero(); accelerator.Synchronize();
                Kernel_CFU_P1(size, cfu.View, ev.View, p1_avg.View, p2_avg.View, mpayoffs.View); accelerator.Synchronize();
                Kernel_CFR(size, p1_cfr.View, cfu.View, ev.View, ic.View); accelerator.Synchronize();
                Kernel_Strats(size, p1_strategy.View, p1_avg.View, ic.View); accelerator.Synchronize();

                // Update player two.
                sums.MemSetToZero(); accelerator.Synchronize();
                Kernel_Positive_Sums(size, sums.View, p1_cfr.View, p2_cfr.View); accelerator.Synchronize();
                Kernel_Avg_Strats(size,  p1_avg.View, p2_avg.View, p1_cfr.View, p2_cfr.View, sums.View); accelerator.Synchronize();
                ev.MemSetToZero(); accelerator.Synchronize();
                Kernel_CFU_P2(size, cfu.View, ev.View, p1_avg.View, p2_avg.View, mpayoffs.View); accelerator.Synchronize();
                Kernel_CFR(size, p2_cfr.View, cfu.View, ev.View, ic.View); accelerator.Synchronize();
                Kernel_Strats(size, p2_strategy.View, p2_avg.View, ic.View); accelerator.Synchronize();

                // Get the Best Responses.
                sums.MemSetToZero(); accelerator.Synchronize();
                brs.CopyFromCPU(new double[] { double.NegativeInfinity, double.NegativeInfinity }); accelerator.Synchronize();
                Kernel_Sums(size, sums.View, p1_strategy.View, p2_strategy.View); accelerator.Synchronize();
                Kernel_Norm_Strats(size, p1_norm.View, p2_norm.View, p1_strategy.View, p2_strategy.View, sums.View); accelerator.Synchronize();
                Kernel_BRs(size, brs.View, p1_norm.View, p2_norm.View, mpayoffs.View); accelerator.Synchronize();

                double[] cb = brs.GetAsArray1D(); accelerator.Synchronize();
                Console.WriteLine(i + ": " + ((cb[0] + cb[1]) / 2));

            }

            accelerator.Synchronize();

            // Show the resulting strategies.
            Console.WriteLine(String.Join(", ", p1_strategy.GetAsArray1D()));
            Console.WriteLine("----------------------------");
            Console.WriteLine(String.Join(", ", p2_strategy.GetAsArray1D()));

            // Clean up.
            accelerator.Dispose();
            context.Dispose();

            p1_cfr.Dispose();
            p2_cfr.Dispose();
            p1_strategy.Dispose();
            p2_strategy.Dispose();
            mpayoffs.Dispose();
            sums.Dispose();
            brs.Dispose();
            ev.Dispose();
            p1_avg.Dispose();
            p2_avg.Dispose();
            p1_norm.Dispose();
            p2_norm.Dispose();
            cfu.Dispose();
            ic.Dispose();

        }

    }

}


