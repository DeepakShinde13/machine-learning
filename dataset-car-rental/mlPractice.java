import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
//import java.lang.*;
public class mlPractice {
	public static void main(String argts[]){
		
		//PythonInterpreter interpreter = new PythonInterpreter();
	   //need to call myscript.py and also pass arg1 as its arguments.
	  //and also myscript.py path is in C:\Demo\myscript.py
		System.out.println("HELLO");
	    String[] terminal = {
	      "python",
	      "/home/deepak/Desktop/ml-predictor.py",
	      "20,3,109813,1,7,4,12,2,4,1,0,0,40,38"
	    };
	    try {
			//System.out.println(Runtime.getRuntime().exec(terminal));
			Runtime rt = Runtime.getRuntime();
			Process proc = rt.exec(terminal);

			BufferedReader stdInput = new BufferedReader(new 
			     InputStreamReader(proc.getInputStream()));

			BufferedReader stdError = new BufferedReader(new 
			     InputStreamReader(proc.getErrorStream()));

			// read the output from the command
			System.out.println("Here is the standard output of the command:\n");
			String s = null;
			while ((s = stdInput.readLine()) != null) {
			    System.out.println(s);
			}

			// read any errors from the attempted command
			System.out.println("Here is the standard error of the command (if any):\n");
			while ((s = stdError.readLine()) != null) {
			    System.out.println(s);
			}
			System.out.println("DONE");
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	    
	}
}
