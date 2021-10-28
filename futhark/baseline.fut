-- Sorting: Radix for baseline
-- ==
-- compiled input { [9i32, 8i32, 7i32, 6i32, 5i32, 4i32, 3i32, 2i32, 1i32] } output { [1i32, 2i32, 3i32, 4i32, 5i32, 6i32, 7i32, 8i32, 9i32] }
import "lib/github.com/diku-dk/sorts/radix_sort"

let sortTest (n : []i32) : []i32 =
    radix_sort i32.num_bits i32.get_bit n


-- RUN a big test with:
-- $ futhark opencl baseline.fut
-- $ echo "[3,2,1]" | ./baseline -t /dev/stderr -r 10 > /dev/null
let main (n : []i32) : []i32 = sortTest n
