package jimple.io;

import java.util.ArrayList;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public final class Common {

	public static final String EmptyString = "";
	public static final String UTF8 = "UTF-8";
	public static final String EvaluateTempDir = "EvalTemp";

	public static final String FieldAccessExpr = "FieldAccessExpr";
	public static final String ClassOrInterfaceType = "ClassOrInterfaceType";
	public static final String MethodDeclaration = "MethodDeclaration";
	public static final String NameExpr = "NameExpr";
	public static final String MethodCallExpr = "MethodCallExpr";
	public static final String DummyNode = "DummyNode";
	public static final String BlankWord = "BLANK";

	public static final int c_MaxLabelLength = 50;
	public static final String methodName = "METHOD_NAME";
	public static final String internalSeparator = "|";

	public static String normalizeName(String original, String defaultString) {
		original = original.toLowerCase().replaceAll("\\\\n", "") // escaped new
																	// lines
				.replaceAll("//s+", "") // whitespaces
				.replaceAll("[\"',]", "") // quotes, apostrophies, commas
				.replaceAll("\\P{Print}", ""); // unicode weird characters
		String stripped = original.replaceAll("[^A-Za-z]", "");
		if (stripped.length() == 0) {
			String carefulStripped = original.replaceAll(" ", "_");
			if (carefulStripped.length() == 0) {
				return defaultString;
			} else {
				return carefulStripped;
			}
		} else {
			return stripped;
		}
	}

	

	public static ArrayList<String> splitToSubtokens(String str1) {
		String str2 = str1.trim();
		return Stream.of(str2.split("(?<=[a-z])(?=[A-Z])|_|[0-9]|(?<=[A-Z])(?=[A-Z][a-z])|\\s+"))
				.filter(s -> s.length() > 0).map(s -> Common.normalizeName(s, Common.EmptyString))
				.filter(s -> s.length() > 0).collect(Collectors.toCollection(ArrayList::new));
	}
}
