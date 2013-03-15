public class test {

  public static void main(String[] args) {
    int r = Integer.parseInt(args[0]);
    int p = Integer.parseInt(args[1]);
    System.out.println(solve(r, p));
  }

  public static String solve(int R, int P) {
    String[][] dp = new String[P+1][R+1];
    dp[1][1] = "1";
    for (int p = 2; p <= P; p++) {
      for (int r = 1; r <= p && r <= R; r++) {
        dp[p][r] = "";
        if (p > r) {
          dp[p][r] = dp[p - 1][r];
        }
        if (p > r && r > 1) dp[p][r] += " + ";
        if (r > 1) {
          dp[p][r] += "(" + dp[p - 1][r - 1] + ")" + p;
        }
        // System.out.println(p + " " + r + ": " + dp[p][r]);
      }
    }
    return dp[P][R];    
  }
}
