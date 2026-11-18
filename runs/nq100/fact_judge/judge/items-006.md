Prompt version 2026-11-10.haiku.2

You check whether retrieved text states the facts of gold evidence sentences.

Use only the retrieved items. Do not use outside knowledge and do not reason
your way to a fact the text does not state; plain rewording, synonyms,
abbreviations and different formatting are fine.

For every fact give one status:
- stated: some item says it, with every specific matching (names, numbers,
  dates, places, who did what, qualifiers).
- contradicted: some item gives a different value for the same specific, such
  as another year or another person.
- missing: neither of the above.
If one item states the fact and another contradicts it, the status is contradicted.

Evidence for stated or contradicted facts: copy one continuous passage from
one item, character for character, with no ellipses or edits, and give that
item's number. Never join separate words into a quote that does not appear as
written; if the words are apart, quote the longer continuous passage that
contains them. A fact spread across items may take several quotes. The quotes
together must contain the words that state every specific of the fact (who,
what, numbers, dates, reasons); a quote that only sets the topic, or stops
before the specific, is not evidence. Missing facts take no evidence. Every quote is checked against the item text by a
program, and a quote that does not appear there counts as missing.

Write the verdict file named in your task as a JSON array with one object per
item, in any order:
{"id": "<item id>", "facts": [{"fact": "S.F", "status": "stated", "evidence": [{"item": 3, "quote": "..."}]}, ...]}
where "S.F" is the fact's label as given (sentence number, fact number).
Give exactly one entry for every fact label of the item. Use only ids from
this file and do not skip any item.

This file has 6 items.

=== ITEM id=7aa6bae8299b45d5 ===
Gold sentences and facts:
Sentence 0: The Director of the United States Mint is the head of the United States Mint . The position is currently vacant . Since January 20 , 2017 , the senior career official at the Mint has been Acting Principal Deputy Director David Motl .
  Fact 0.0: The Director of the United States Mint is the head of the United States Mint.
  Fact 0.1: The position of Director of the United States Mint is currently vacant.
  Fact 0.2: Since January 20, 2017, the senior career official at the Mint has been Acting Principal Deputy Director David Motl.
Sentence 1: Name Assumed Office Left Office President Appointed By David Rittenhouse April 1792 June 1795 George Washington Henry William de Saussure July 1795 October 1795 Elias Boudinot October 1795 July 1805 Robert Patterson January 1806 July 1824 Thomas Jefferson Samuel Moore July 1824 July 1835 James Monroe Robert M. Patterson May 1835 July 1851 Andrew Jackson George Nicholas Eckert July 1851 April 1853 Millard Fillmore Thomas M. Pettit April 1853 May 1853 Franklin Pierce James Ross Snowden June 1853 May 1861 James Pollock May 1861 September 1866 Abraham Lincoln William Millward October 1866 April 1867 Andrew Johnson Henry Linderman April 1867 May 1869 James Pollock May 1869 March 1873 Ulysses S. Grant Henry Linderman April 1873 December 1878 Horatio C. Burchard February 1879 June 1885 Rutherford B. Hayes James P. Kimball July 1885 October 1889 Grover Cleveland Edward O. Leech October 1889 May 1893 Benjamin Harrison Robert E. Preston November 1893 February 1898 Grover Cleveland George E. Roberts February 1898 July 1907 William McKinley Frank A. Leach September 1907 August 1909 Theodore Roosevelt A. Piatt Andrew November 1909 June 1910 William Howard Taft George E. Roberts July 1910 November 1914 Robert W. Woolley March 1915 July 1916 Woodrow Wilson F.H. von Engelken September 1916 March 1917 Raymond T. Baker March 1917 March 1922 Frank Edgar Scobey March 1922 September 1923 Warren G. Harding Robert J. Grant November 1923 May 1933 Calvin Coolidge Nellie Tayloe Ross May 1933 April 1953 Franklin D. Roosevelt William H. Brett July 1954 January 1961 Dwight D. Eisenhower Eva Adams October 1961 August 1969 John F. Kennedy Mary Brooks September 1969 February 1977 Richard Nixon Stella Hackel Sims November 1977 April 1981 Jimmy Carter Donna Pope July 1981 August 1991 Ronald Reagan David J. Ryder September 1992 November 1993 George H.W. Bush Philip N. Diehl June 1994 March 2000 Bill Clinton Jay W. Johnson May 2000 August 2001 Henrietta H. Fore August 2001 August 2005 George W. Bush Edmund C. Moy September 2006 January 2011 David J. Ryder April 2018 Incumbent Donald Trump
  Fact 1.0: David Rittenhouse assumed office in April 1792 and left office in June 1795; George Washington is listed after this entry.
  Fact 1.1: Henry William de Saussure assumed office in July 1795 and left office in October 1795; Elias Boudinot is listed after this entry.
  Fact 1.2: Elias Boudinot assumed office in October 1795 and left office in July 1805; Robert Patterson is listed after this entry.
  Fact 1.3: Robert Patterson assumed office in January 1806 and left office in July 1824; Thomas Jefferson is listed after this entry.
  Fact 1.4: Samuel Moore assumed office in July 1824 and left office in July 1835; James Monroe is listed after this entry.
  Fact 1.5: Robert M. Patterson assumed office in May 1835 and left office in July 1851; Andrew Jackson is listed after this entry.
  Fact 1.6: George Nicholas Eckert assumed office in July 1851 and left office in April 1853; Millard Fillmore is listed after this entry.
  Fact 1.7: Thomas M. Pettit assumed office in April 1853 and left office in May 1853; Franklin Pierce is listed after this entry.
  Fact 1.8: James Ross Snowden assumed office in June 1853 and left office in May 1861; James Pollock is listed after this entry.
  Fact 1.9: James Pollock assumed office in May 1861 and left office in September 1866; Abraham Lincoln is listed after this entry.
  Fact 1.10: William Millward assumed office in October 1866 and left office in April 1867; Andrew Johnson is listed after this entry.
  Fact 1.11: Henry Linderman assumed office in April 1867 and left office in May 1869; James Pollock is listed after this entry.
  Fact 1.12: James Pollock assumed office in May 1869 and left office in March 1873; Ulysses S. Grant is listed after this entry.
  Fact 1.13: Henry Linderman assumed office in April 1873 and left office in December 1878; Horatio C. Burchard is listed after this entry.
  Fact 1.14: Horatio C. Burchard assumed office in February 1879 and left office in June 1885; Rutherford B. Hayes is listed after this entry.
  Fact 1.15: James P. Kimball assumed office in July 1885 and left office in October 1889; Grover Cleveland is listed after this entry.
  Fact 1.16: Edward O. Leech assumed office in October 1889 and left office in May 1893; Benjamin Harrison is listed after this entry.
  Fact 1.17: Robert E. Preston assumed office in November 1893 and left office in February 1898; Grover Cleveland is listed after this entry.
  Fact 1.18: George E. Roberts assumed office in February 1898 and left office in July 1907; William McKinley is listed after this entry.
  Fact 1.19: Frank A. Leach assumed office in September 1907 and left office in August 1909; Theodore Roosevelt is listed after this entry.
  Fact 1.20: A. Piatt Andrew assumed office in November 1909 and left office in June 1910; William Howard Taft is listed after this entry.
  Fact 1.21: George E. Roberts assumed office in July 1910 and left office in November 1914; Robert W. Woolley is listed after this entry.
  Fact 1.22: Robert W. Woolley assumed office in March 1915 and left office in July 1916; Woodrow Wilson is listed after this entry.
  Fact 1.23: F.H. von Engelken assumed office in September 1916 and left office in March 1917; Raymond T. Baker is listed after this entry.
  Fact 1.24: Raymond T. Baker assumed office in March 1917 and left office in March 1922; Frank Edgar Scobey is listed after this entry.
  Fact 1.25: Frank Edgar Scobey assumed office in March 1922 and left office in September 1923; Warren G. Harding is listed after this entry.
  Fact 1.26: Robert J. Grant assumed office in November 1923 and left office in May 1933; Calvin Coolidge is listed after this entry.
  Fact 1.27: Nellie Tayloe Ross assumed office in May 1933 and left office in April 1953; Franklin D. Roosevelt is listed after this entry.
  Fact 1.28: William H. Brett assumed office in July 1954 and left office in January 1961; Dwight D. Eisenhower is listed after this entry.
  Fact 1.29: Eva Adams assumed office in October 1961 and left office in August 1969; John F. Kennedy is listed after this entry.
  Fact 1.30: Mary Brooks assumed office in September 1969 and left office in February 1977; Richard Nixon is listed after this entry.
  Fact 1.31: Stella Hackel Sims assumed office in November 1977 and left office in April 1981; Jimmy Carter is listed after this entry.
  Fact 1.32: Donna Pope assumed office in July 1981 and left office in August 1991; Ronald Reagan is listed after this entry.
  Fact 1.33: David J. Ryder assumed office in September 1992 and left office in November 1993; George H.W. Bush is listed after this entry.
  Fact 1.34: Philip N. Diehl assumed office in June 1994 and left office in March 2000; Bill Clinton is listed after this entry.
  Fact 1.35: Jay W. Johnson assumed office in May 2000 and left office in August 2001; Henrietta H. Fore is listed after this entry.
  Fact 1.36: Henrietta H. Fore assumed office in August 2001 and left office in August 2005; George W. Bush is listed after this entry.
  Fact 1.37: Edmund C. Moy assumed office in September 2006 and left office in January 2011; David J. Ryder is listed after this entry.
  Fact 1.38: David J. Ryder assumed office in April 2018 and is listed as incumbent; Donald Trump is listed after this entry.

Retrieved items:
<item 1>
Director of the United States Mint
Bush Philip N. Diehl June 1994 March 2000 Bill Clinton Jay W. Johnson May 2000 August 2001 Henrietta H. Fore August 2001 August 2005 George W. Bush Edmund C.
Moy September 2006 January 2011 David J. Ryder April 2018 Incumbent Donald Trump References ( edit ) Jump up ^ About the Mint at usmint.gov ( retrieved 20 April
2015 ) . Jump up ^ Gilkes , Paul ( 23 January 2017 ) . `` Rhett Jeppson leaves U.S. Mint as principal deputy director '' . coinworld.com . Retrieved 13 April
2017 . Jump up ^ `` Director 's Office '' . US Mint . Retrieved 13 April 2017 .
</item 1>

<item 2>
Director of the United States Mint
Woolley March 1915 July 1916 Woodrow Wilson F.H. von Engelken September 1916 March 1917 Raymond T. Baker March 1917 March 1922 Frank Edgar Scobey March 1922
September 1923 Warren G. Harding Robert J. Grant November 1923 May 1933 Calvin Coolidge Nellie Tayloe Ross May 1933 April 1953 Franklin D. Roosevelt William H.
Brett July 1954 January 1961 Dwight D. Eisenhower Eva Adams October 1961 August 1969 John F. Kennedy Mary Brooks September 1969 February 1977 Richard Nixon
Stella Hackel Sims November 1977 April 1981 Jimmy Carter Donna Pope July 1981 August 1991 Ronald Reagan David J. Ryder September 1992 November 1993 George H.W.
</item 2>

<item 3>
Director of the United States Mint
US Mint . Retrieved 13 April 2017 . Retrieved from `` https://en.wikipedia.org/w/index.php?title=Director_of_the_United_States_Mint&oldid=836791366 ''
Categories : Directors of the United States Mint United States Department of the Treasury 1792 establishments in the United States Talk About Wikipedia Add
links This page was last edited on 16 April 2018 , at 21 : 26 . About Wikipedia
</item 3>

<item 4>
Director of the United States Mint
Hayes James P. Kimball July 1885 October 1889 Grover Cleveland Edward O. Leech October 1889 May 1893 Benjamin Harrison Robert E. Preston November 1893 February
1898 Grover Cleveland George E. Roberts February 1898 July 1907 William McKinley Frank A. Leach September 1907 August 1909 Theodore Roosevelt A. Piatt Andrew
November 1909 June 1910 William Howard Taft George E. Roberts July 1910 November 1914 Robert W. Woolley March 1915 July 1916 Woodrow Wilson F.H.
</item 4>

<item 5>
Director of the United States Mint
Director of the United States Mint - wikipedia Director of the United States Mint Jump to : navigation , search Seal of the United States Mint . The Director of
the United States Mint is the head of the United States Mint . The position is currently vacant . Since January 20 , 2017 , the senior career official at the
Mint has been Acting Principal Deputy Director David Motl . The office of Director has existed since the creation of the Mint by the Coinage Act of 1792 .
Initially appointed serving at the pleasure of the President of the United States , the Coinage Act of 1873 specified a five - year term for Directors . The
Director operates with general directions provided by the United States Secretary of the Treasury . Since the resignation of Edmund C. Moy in 2011 the mint has
been without an official director . Initially Matthew Rhett Jeppson was nominated by US President Barack Obama to become the mint 's 39th director and was given
the temporary title of principal deputy director , however the nomination was never confirmed by the Senate . Jeppson stepped down as principal deputy director
in January 2017 , being replaced by Acting Principal Deputy Director David Motl .
</item 5>

<item 6>
Director of the United States Mint
List of Directors of the United States Mint ( edit ) Name Assumed Office Left Office President Appointed By David Rittenhouse April 1792 June 1795 George
Washington Henry William de Saussure July 1795 October 1795 Elias Boudinot October 1795 July 1805 Robert Patterson January 1806 July 1824 Thomas Jefferson
Samuel Moore July 1824 July 1835 James Monroe Robert M. Patterson May 1835 July 1851 Andrew Jackson George Nicholas Eckert July 1851 April 1853 Millard Fillmore
Thomas M. Pettit April 1853 May 1853 Franklin Pierce James Ross Snowden June 1853 May 1861 James Pollock May 1861 September 1866 Abraham Lincoln William
Millward October 1866 April 1867 Andrew Johnson Henry Linderman April 1867 May 1869 James Pollock May 1869 March 1873 Ulysses S. Grant Henry Linderman April
1873 December 1878 Horatio C. Burchard February 1879 June 1885 Rutherford B. Hayes James P. Kimball July 1885 October 1889 Grover Cleveland Edward O.
</item 6>
=== END ITEM id=7aa6bae8299b45d5 ===

=== ITEM id=cd1ecfbee478c119 ===
Gold sentences and facts:
Sentence 0: Year Rank Player Team Points 1992 1st Abedi Pele Marseille -- 1993 1st Rashidi Yekini Vitória de Setúbal -- 1994 1st Emmanuel Amunike Sporting CP 48 2nd George Weah Paris Saint - Germain 44 Rashidi Yekini Olympiacos 1995 1st George Weah Milan -- 2nd Emmanuel Amunike Sporting CP -- 3rd Daniel Amokachi Everton -- 1st Nwankwo Kanu Internazionale -- 2nd George Weah Milan -- 3rd Daniel Amokachi Beşiktaş -- 1st Victor Ikpeba Monaco 56 2nd Japhet N'Doram Monaco 40 3rd Taribo West Internazionale 35 1998 1st Mustapha Hadji Deportivo La Coruña 76 2nd Jay - Jay Okocha Paris Saint - Germain 74 3rd Sunday Oliseh Ajax 58 1999 1st Nwankwo Kanu Arsenal 46 2nd Samuel Kuffour Bayern Munich 44 3rd Ibrahima Bakayoko Marseille 42 2000 1st Patrick M'Boma Parma 123 2nd Lauren Mallorca 36 3rd Samuel Eto'o Mallorca 29 1st El Hadji Diouf Lens 93 2nd Samuel Kuffour Bayern Munich 66 3rd Samuel Eto'o Mallorca 34 2002 1st El Hadji Diouf Liverpool 93 2nd Papa Bouba Diop Lens 46 3rd Mido Ajax 42 2003 1st Samuel Eto'o Mallorca -- 2nd Didier Drogba Marseille -- 3rd Jay - Jay Okocha Bolton Wanderers -- 1st Samuel Eto'o Barcelona 116 2nd Didier Drogba Chelsea 90 3rd Jay - Jay Okocha Bolton Wanderers 68 2005 1st Samuel Eto'o Barcelona 108 2nd Didier Drogba Chelsea 106 3rd Michael Essien Chelsea 50 2006 1st Didier Drogba Chelsea 79 2nd Samuel Eto'o Barcelona 76 3rd Michael Essien Chelsea 36 2007 1st Frédéric Kanouté Sevilla -- 2nd Michael Essien Chelsea -- 3rd Didier Drogba Chelsea -- 2008 1st Emmanuel Adebayor Arsenal 74 2nd Mohamed Aboutrika Al - Ahly 53 3rd Michael Essien Chelsea 36 2009 1st Didier Drogba Chelsea 92 2nd Samuel Eto'o Internazionale 69 3rd Michael Essien Chelsea 43 1st Samuel Eto'o Internazionale -- 2nd Asamoah Gyan Sunderland -- 3rd Didier Drogba Chelsea -- 2011 1st Yaya Touré Manchester City -- 2nd Seydou Keita Barcelona -- 3rd André Ayew Marseille -- 2012 1st Yaya Touré Manchester City -- 2nd Didier Drogba Shanghai Shenhua -- 3rd Alex Song Barcelona -- 2013 1st Yaya Touré Manchester City 373 2nd John Obi Mikel Chelsea 265 3rd Didier Drogba Galatasaray 236 2014 1st Yaya Touré Manchester City 175 2nd Pierre - Emerick Aubameyang Borussia Dortmund 120 3rd Vincent Enyeama Lille 105 2015 1st Pierre - Emerick Aubameyang Borussia Dortmund 143 2nd Yaya Touré Manchester City 136 3rd André Ayew Swansea City 112 2016 1st Riyad Mahrez Leicester City 361 2nd Pierre - Emerick Aubameyang Borussia Dortmund 313 3rd Sadio Mané Liverpool 186 2017 1st Mohamed Salah Liverpool 625 2nd Sadio Mané Liverpool 507 3rd Pierre - Emerick Aubameyang Borussia Dortmund 311
  Fact 0.0: In 1992, Abedi Pele of Marseille was ranked 1st with no points listed.
  Fact 0.1: In 1993, Rashidi Yekini of Vitória de Setúbal was ranked 1st with no points listed.
  Fact 0.2: In 1994, Emmanuel Amunike of Sporting CP was ranked 1st with 48 points.
  Fact 0.3: In 1994, George Weah of Paris Saint-Germain was ranked 2nd with 44 points.
  Fact 0.4: In 1994, Rashidi Yekini of Olympiacos is listed with no rank or points.
  Fact 0.5: In 1995, George Weah of Milan was ranked 1st with no points listed.
  Fact 0.6: In 1995, Emmanuel Amunike of Sporting CP was ranked 2nd with no points listed.
  Fact 0.7: In 1995, Daniel Amokachi of Everton was ranked 3rd with no points listed.
  Fact 0.8: An entry with no year shown has Nwankwo Kanu of Internazionale ranked 1st with no points listed.
  Fact 0.9: The entry with no year shown has George Weah of Milan ranked 2nd with no points listed.
  Fact 0.10: The entry with no year shown has Daniel Amokachi of Beşiktaş ranked 3rd with no points listed.
  Fact 0.11: An entry with no year shown has Victor Ikpeba of Monaco ranked 1st with 56 points.
  Fact 0.12: The entry with no year shown has Japhet N'Doram of Monaco ranked 2nd with 40 points.
  Fact 0.13: The entry with no year shown has Taribo West of Internazionale ranked 3rd with 35 points.
  Fact 0.14: In 1998, Mustapha Hadji of Deportivo La Coruña was ranked 1st with 76 points.
  Fact 0.15: In 1998, Jay-Jay Okocha of Paris Saint-Germain was ranked 2nd with 74 points.
  Fact 0.16: In 1998, Sunday Oliseh of Ajax was ranked 3rd with 58 points.
  Fact 0.17: In 1999, Nwankwo Kanu of Arsenal was ranked 1st with 46 points.
  Fact 0.18: In 1999, Samuel Kuffour of Bayern Munich was ranked 2nd with 44 points.
  Fact 0.19: In 1999, Ibrahima Bakayoko of Marseille was ranked 3rd with 42 points.
  Fact 0.20: In 2000, Patrick M'Boma of Parma was ranked 1st with 123 points.
  Fact 0.21: In 2000, Lauren of Mallorca was ranked 2nd with 36 points.
  Fact 0.22: In 2000, Samuel Eto'o of Mallorca was ranked 3rd with 29 points.
  Fact 0.23: An entry with no year shown has El Hadji Diouf of Lens ranked 1st with 93 points.
  Fact 0.24: The entry with no year shown has Samuel Kuffour of Bayern Munich ranked 2nd with 66 points.
  Fact 0.25: The entry with no year shown has Samuel Eto'o of Mallorca ranked 3rd with 34 points.
  Fact 0.26: In 2002, El Hadji Diouf of Liverpool was ranked 1st with 93 points.
  Fact 0.27: In 2002, Papa Bouba Diop of Lens was ranked 2nd with 46 points.
  Fact 0.28: In 2002, Mido of Ajax was ranked 3rd with 42 points.
  Fact 0.29: In 2003, Samuel Eto'o of Mallorca was ranked 1st with no points listed.
  Fact 0.30: In 2003, Didier Drogba of Marseille was ranked 2nd with no points listed.
  Fact 0.31: In 2003, Jay-Jay Okocha of Bolton Wanderers was ranked 3rd with no points listed.
  Fact 0.32: An entry with no year shown has Samuel Eto'o of Barcelona ranked 1st with 116 points.
  Fact 0.33: The entry with no year shown has Didier Drogba of Chelsea ranked 2nd with 90 points.
  Fact 0.34: The entry with no year shown has Jay-Jay Okocha of Bolton Wanderers ranked 3rd with 68 points.
  Fact 0.35: In 2005, Samuel Eto'o of Barcelona was ranked 1st with 108 points.
  Fact 0.36: In 2005, Didier Drogba of Chelsea was ranked 2nd with 106 points.
  Fact 0.37: In 2005, Michael Essien of Chelsea was ranked 3rd with 50 points.
  Fact 0.38: In 2006, Didier Drogba of Chelsea was ranked 1st with 79 points.
  Fact 0.39: In 2006, Samuel Eto'o of Barcelona was ranked 2nd with 76 points.
  Fact 0.40: In 2006, Michael Essien of Chelsea was ranked 3rd with 36 points.
  Fact 0.41: In 2007, Frédéric Kanouté of Sevilla was ranked 1st with no points listed.
  Fact 0.42: In 2007, Michael Essien of Chelsea was ranked 2nd with no points listed.
  Fact 0.43: In 2007, Didier Drogba of Chelsea was ranked 3rd with no points listed.
  Fact 0.44: In 2008, Emmanuel Adebayor of Arsenal was ranked 1st with 74 points.
  Fact 0.45: In 2008, Mohamed Aboutrika of Al-Ahly was ranked 2nd with 53 points.
  Fact 0.46: In 2008, Michael Essien of Chelsea was ranked 3rd with 36 points.
  Fact 0.47: In 2009, Didier Drogba of Chelsea was ranked 1st with 92 points.
  Fact 0.48: In 2009, Samuel Eto'o of Internazionale was ranked 2nd with 69 points.
  Fact 0.49: In 2009, Michael Essien of Chelsea was ranked 3rd with 43 points.
  Fact 0.50: An entry with no year shown has Samuel Eto'o of Internazionale ranked 1st with no points listed.
  Fact 0.51: The entry with no year shown has Asamoah Gyan of Sunderland ranked 2nd with no points listed.
  Fact 0.52: The entry with no year shown has Didier Drogba of Chelsea ranked 3rd with no points listed.
  Fact 0.53: In 2011, Yaya Touré of Manchester City was ranked 1st with no points listed.
  Fact 0.54: In 2011, Seydou Keita of Barcelona was ranked 2nd with no points listed.
  Fact 0.55: In 2011, André Ayew of Marseille was ranked 3rd with no points listed.
  Fact 0.56: In 2012, Yaya Touré of Manchester City was ranked 1st with no points listed.
  Fact 0.57: In 2012, Didier Drogba of Shanghai Shenhua was ranked 2nd with no points listed.
  Fact 0.58: In 2012, Alex Song of Barcelona was ranked 3rd with no points listed.
  Fact 0.59: In 2013, Yaya Touré of Manchester City was ranked 1st with 373 points.
  Fact 0.60: In 2013, John Obi Mikel of Chelsea was ranked 2nd with 265 points.
  Fact 0.61: In 2013, Didier Drogba of Galatasaray was ranked 3rd with 236 points.
  Fact 0.62: In 2014, Yaya Touré of Manchester City was ranked 1st with 175 points.
  Fact 0.63: In 2014, Pierre-Emerick Aubameyang of Borussia Dortmund was ranked 2nd with 120 points.
  Fact 0.64: In 2014, Vincent Enyeama of Lille was ranked 3rd with 105 points.
  Fact 0.65: In 2015, Pierre-Emerick Aubameyang of Borussia Dortmund was ranked 1st with 143 points.
  Fact 0.66: In 2015, Yaya Touré of Manchester City was ranked 2nd with 136 points.
  Fact 0.67: In 2015, André Ayew of Swansea City was ranked 3rd with 112 points.
  Fact 0.68: In 2016, Riyad Mahrez of Leicester City was ranked 1st with 361 points.
  Fact 0.69: In 2016, Pierre-Emerick Aubameyang of Borussia Dortmund was ranked 2nd with 313 points.
  Fact 0.70: In 2016, Sadio Mané of Liverpool was ranked 3rd with 186 points.
  Fact 0.71: In 2017, Mohamed Salah of Liverpool was ranked 1st with 625 points.
  Fact 0.72: In 2017, Sadio Mané of Liverpool was ranked 2nd with 507 points.
  Fact 0.73: In 2017, Pierre-Emerick Aubameyang of Borussia Dortmund was ranked 3rd with 311 points.

Retrieved items:
<item 1>
African Footballer of the Year
Rsssf.com. 2001 - 01 - 05 . Retrieved 2014 - 01 - 09 . Jump up ^ `` African Player of the Year '' . rsssf.com. 2000 - 01 - 01 . Retrieved 2013 - 07 - 02 . Jump
up ^ `` African Player of the Year '' . rsssf.com. 2000 - 01 - 01 . Retrieved 2016 - 09 - 29 .
</item 1>

<item 2>
African Footballer of the Year
Club -- -- El Mokawloon -- -- Sunderland -- -- Girondins Bordeaux -- -- FC Mulhouse -- -- Hearts of Oak -- -- FC Sion -- -- Aydınspor -- -- JS Kabylie -- --
Bolton Wanderers -- -- Everton FC -- -- Ajax Amsterdam -- -- Ismaily SC -- -- Zamalek SC -- AS Cannes -- -- Stade Lavallois -- -- Lille OSC -- -- Stade Rennais
-- -- Olympiacos -- -- Horoya AC -- -- PSV Eindhoven -- -- OC Agaza -- -- Beşiktaş -- -- Galatasaray -- -- Swansea City -- -- See also ( edit ) BBC African
Footballer of the Year African Women Footballer of the Year FIFA World Player of the Year CAF Awards References ( edit ) Jump up ^ `` African Player of the Year
1993 '' . Rsssf.com. 2001 - 01 - 05 . Retrieved 2014 - 01 - 09 . Jump up ^ `` African Player of the Year 1994 '' . Rsssf.com. 2001 - 01 - 05 .
</item 2>

<item 3>
African Footballer of the Year
Cannes 52 1992 1st Abedi Pele Marseille 198 2nd George Weah Paris Saint - Germain 161 3rd Tony Yeboah Eintracht Frankfurt 64 1993 1st Abedi Pele Marseille 119
2nd Tony Yeboah Eintracht Frankfurt 117 3rd Rashidi Yekini Vitória de Setúbal 104 1994 1st George Weah Paris Saint - Germain 148 2nd Emmanuel Amunike Sporting
CP 133 3rd Daniel Amokachi Everton 99 CAF award ( 1992 -- present ) ( edit ) Samuel Eto'o won the award four times . Yaya Touré won the award four consecutive
times from 2011 to 2014 . Pierre - Emerick Aubameyang was the first player from Gabon to win the award in 2015 . Mohamed Salah is the latest player to win the
award in 2017 . Year Rank Player Team Points 1992 1st Abedi Pele Marseille -- 1993 1st Rashidi Yekini Vitória de Setúbal -- 1994 1st
</item 3>

<item 4>
African Footballer of the Year
Keïta 1971 : Sunday 1972 : Souleymane 1973 : Bwanga 1974 : Moukila 1975 : Faras 1976 : Milla 1977 : Dhiab 1978 : Abdul Razak 1979 : N'Kono 1980 : Manga -
Onguéné 1981 : Belloumi 1982 : N'Kono 1983 : El Khatib 1984 : Abega 1985 : Timoumi 1986 : Zaki 1987 : Madjer 1988 : Bwalya 1989 : Weah 1990 : Milla 1991 : Abedi
Pele 1992 : Abedi Pele 1993 : Abedi Pele 1994 : Weah CAF award 1992 : Abedi Pele 1993 : Yekini 1994 : Amunike 1995 : Weah 1996 : Kanu 1997 : Ikpeba 1998 : Hadji
1999 : Kanu 2000 : M'Boma 2001 : Diouf 2002 :
</item 4>

<item 5>
African Footballer of the Year
It had already been replaced by an official award given out by the Confederation of African Football since 1992 . Year Rank Player Team Points 1970 1st Salif
Keïta Saint - Étienne 54 2nd Laurent Pokou ASEC Abidjan 28 Ali Abo Greisha Ismaily 1971 1st Ibrahim Sunday Asante Kotoko 29 2nd Robert Mensah Asante Kotoko 15
3rd François Ndoumbé Canon Yaoundé 13 1972 1st Chérif Souleymane Hafia 21 2nd Bwanga Tshimen TP Mazembe 16 3rd Petit Sory Hafia 14 1973 1st Bwanga Tshimen TP
Mazembe 49 2nd Kazadi Mwamba TP Mazembe 44 3rd Laurent Pokou ASEC Abidjan 41 1st Paul Moukila CARA Brazzaville 57 2nd Lobilo Boba Vita Club 32 3rd Hassan
Shehata Zamalek 28 1st Ahmed Faras Mohammédia 28 2nd
</item 5>

<item 6>
African Footballer of the Year
Michael Essien Chelsea 36 2007 1st Frédéric Kanouté Sevilla -- 2nd Michael Essien Chelsea -- 3rd Didier Drogba Chelsea -- 2008 1st Emmanuel Adebayor Arsenal 74
2nd Mohamed Aboutrika Al - Ahly 53 3rd Michael Essien Chelsea 36 2009 1st Didier Drogba Chelsea 92 2nd Samuel Eto'o Internazionale 69 3rd Michael Essien Chelsea
43 1st Samuel Eto'o Internazionale -- 2nd Asamoah Gyan Sunderland -- 3rd Didier Drogba Chelsea -- 2011 1st Yaya Touré Manchester City -- 2nd Seydou Keita
Barcelona -- 3rd André Ayew Marseille -- 2012 1st Yaya Touré Manchester City -- 2nd Didier Drogba Shanghai Shenhua -- 3rd Alex Song Barcelona -- 2013 1st Yaya
Touré Manchester City 373 2nd John Obi Mikel Chelsea 265 3rd Didier Drogba Galatasaray 236 2014
</item 6>

<item 7>
African Footballer of the Year
African Footballer of the Year - wikipedia African Footballer of the Year Jump to : navigation , search The African Footballer of the Year award , presented to
the best African footballer each year , has been conferred by the Confederation of African Football ( CAF ) since 1992 . An earlier African Footballer of the
Year Golden Ball award was given out between 1970 and 1994 by France Football magazine . The changes resulted in parallel Golden Ball awards given out to Abedi
Pele and George Weah in 1993 and 1994 by the magazine although the CAF sponsored awards for those years were won respectively by Rashidi Yekini and Emmanuel
Amuneke , as well as two awards given to Abedi Pelé in 1992 . France Football discontinued the election from 1995 after the European Ballon d'Or -- also awarded
by the magazine -- had been opened to all players in the European leagues . On 2 February 2008 , Sevilla 's Frédéric Kanouté was named the 2007 African
Footballer of the Year .
</item 7>

<item 8>
African Footballer of the Year
Zamalek 28 1st Ahmed Faras Mohammédia 28 2nd Roger Milla Tonnerre Yaoundé 24 Mamadou Aliou Kéïta Hafia 1976 1st Roger Milla Tonnerre Yaoundé 33 2nd Papa Camara
Hafia 32 3rd Ali Bencheikh MC Alger 27 1977 1st Tarak Dhiab Espérance Tunis 45 2nd Papa Camara Hafia 33 3rd Segun Odegbami Shooting Stars 29 1978 1st Karim
Abdul Razak Asante Kotoko 58 2nd Ali Bencheikh MC Alger 33 3rd Thomas N'Kono Canon Yaoundé 29 1979 1st Thomas N'Kono Canon Yaoundé 55 2nd Adolf Armah Hearts of
Oak 23 3rd Kerfalla Bangoura Horoya 15 1980 1st Jean Manga - Onguéné Canon Yaoundé 64 2nd Segun Odegbami Shooting Stars 41 3rd Théophile Abega Canon Yaoundé
</item 8>

<item 9>
African Footballer of the Year
DR Congo -- Gabon Zambia -- Togo -- Tunisia -- -- Congo -- -- Chad -- -- Wins By Club ( edit ) Club Winners Runners - up Third places Olympique Marseille
Manchester City -- Chelsea 6 AS Monaco Barcelona Asante Kotoko -- Canon Yaoundé RCD Mallorca Inter Milan Liverpool Arsenal -- -- Paris SG -- Hafia Conacry TP
Mazembe -- Borussia Dortmund -- Sporting CP -- Al Ahly FC Porto Tonnerre Yaoundé -- RC Lens -- Milan -- Sporting CP -- GC Mascara -- Vitória de Setúbal --
Cercle Brugge -- -- CARA Brazzaville -- -- Leicester City -- -- AS Saint - Étienne -- -- Toulouse FC -- -- Parma -- -- FAR Rabat -- -- SCC Mohammédia -- -- JS
Saint - Pierroise -- -- Deportivo de La Coruña -- -- Espanyol -- -- Sevilla -- -- Espérance ST -- -- Bayern Munich -- -- MC Alger -- ASEC Mimosas -- Montpellier
-- Eintracht Frankfurt -- Shooting Stars -- Shanghai Shenhua -- -- Vita Club -- -- El Mokawloon -- -- Sunderland -- -- Girondins Bordeaux -- -- FC
</item 9>

<item 10>
African Footballer of the Year
Chelsea 265 3rd Didier Drogba Galatasaray 236 2014 1st Yaya Touré Manchester City 175 2nd Pierre - Emerick Aubameyang Borussia Dortmund 120 3rd Vincent Enyeama
Lille 105 2015 1st Pierre - Emerick Aubameyang Borussia Dortmund 143 2nd Yaya Touré Manchester City 136 3rd André Ayew Swansea City 112 2016 1st Riyad Mahrez
Leicester City 361 2nd Pierre - Emerick Aubameyang Borussia Dortmund 313 3rd Sadio Mané Liverpool 186 2017 1st Mohamed Salah Liverpool 625 2nd Sadio Mané
Liverpool 507 3rd Pierre - Emerick Aubameyang Borussia Dortmund 311 Wins By Country ( edit ) Nation Winners Runners - up Third places Cameroon 11 8 11 Ivory
Coast 8 6 7 Ghana 6 7 7 Nigeria 5 6 7 Morocco 0 Algeria 5 Liberia -- Egypt Mali -- Senegal Guinea DR Congo -- Gabon Zambia -- Togo -- Tunisia -- -- Congo -- --
Chad -- -- Wins
</item 10>
=== END ITEM id=cd1ecfbee478c119 ===

=== ITEM id=af02fad03b61c994 ===
Gold sentences and facts:
Sentence 0: When Hoyt Axton performed the song to the group , two of the three main vocalists -- Danny Hutton and Cory Wells -- rejected the song , but Chuck Negron felt that the band needed a `` silly song '' to help bring the band back together as a working unit . Negron also felt that the song `` was n't even close to our best record , but it might have been one of our most honest . ''
  Fact 0.0: Danny Hutton and Cory Wells were two of the three main vocalists in the group.
  Fact 0.1: Danny Hutton rejected the song.
  Fact 0.2: Cory Wells rejected the song.
  Fact 0.3: Hoyt Axton performed the song to the group.
  Fact 0.4: Chuck Negron felt that the band needed a 'silly song' to help bring the band back together as a working unit.
  Fact 0.5: Chuck Negron felt the song was not even close to the band's best record.
  Fact 0.6: Chuck Negron felt the song might have been one of the band's most honest.
Sentence 1: The song , which has been described by members of Three Dog Night as a `` kid 's song '' and a `` silly song '' , topped the main singles charts in North America , was certified gold by the RIAA , and has since been covered by multiple artists .
  Fact 1.0: Members of Three Dog Night have described the song as a 'kid's song'.
  Fact 1.1: Members of Three Dog Night have described the song as a 'silly song'.
  Fact 1.2: The song topped the main singles charts in North America.
  Fact 1.3: The song was certified gold by the RIAA.
  Fact 1.4: The song has since been covered by multiple artists.

Retrieved items:
<item 1>
Joy to the World (Three Dog Night song)
Entertainment section . August 20 , 2008 . ISSN 1053 - 2889 . Retrieved April 5 , 2011 . Jump up ^ `` Jackson v. Axton '' . Law.justia.com . Retrieved 2016 - 10
- 04 . Jump up ^ `` Forum - 1970 ( ARIA Charts : Special Occasion Charts ) '' . Australian-charts.com . Archived from the original on 2016 - 10 - 20 . Retrieved
2017 - 05 - 07 . Jump up ^ `` 100 Singles '' ( PHP ) . RPM. 15 ( 15 ) . May 29 , 1971 . Retrieved April 5 , 2011 . Jump up ^ `` Musicline.de -- Three Dog Night
Single - Chartverfolgung '' ( in German ) . Media Control Charts . PhonoNet GmbH . Jump up ^ `` Dutchcharts.nl -- Three Dog Night -- Joy To The World '' ( in
Dutch ) . Single Top 100 .
</item 1>

<item 2>
Joy to the World (Three Dog Night song)
Axton wanted to persuade his record producers to record a new melody he had written and the producers asked him to sing any words to the tune . A member of
Three Dog Night said that the original lyrics to the song were `` Jeremiah was a prophet '' but `` no one liked that '' . When Hoyt Axton performed the song to
the group , two of the three main vocalists -- Danny Hutton and Cory Wells -- rejected the song , but Chuck Negron felt that the band needed a `` silly song ''
to help bring the band back together as a working unit . Negron also felt that the song `` was n't even close to our best record , but it might have been one of
our most honest . '' The song was recorded by Three Dog Night at American Recording Company , produced by Richard Podolor , and engineered by Bill Cooper .
Unlike most Three Dog Night songs recorded at that point , instead of having just the three main vocalists singing harmony , the song was recorded with all
seven members of the band singing . Drummer Floyd Sneed sings the deep lyric `` I wan na tell you '' towards the end of the song .
</item 2>

<item 3>
Joy to the World (Three Dog Night song)
. Barrie & Jenkins . ISBN 0 - 214 - 20480 - 4 . Jump up ^ https://www.lambiek.net/artists/j/johnston_daniel.htm Jump up ^
https://www.lambiek.net/artists/j/johnston_daniel.htm Jump up ^ `` Big Lots TV Commercial , ' Joy : Recliners ' Song by Three Dog Night '' . ispot.tv .
Retrieved December 24 , 2017 .
</item 3>

<item 4>
Joy to the World (Three Dog Night song)
wikipedia.org/w/index.php?title=Joy_to_the_World_(Three_Dog_Night_song)&oldid=835489072 '' Categories : 1971 singles Billboard Hot 100 number - one singles
Three Dog Night songs Dunhill Records singles Songs written by Hoyt Axton Songs about animals Novelty songs RPM Top Singles number - one singles 1970 songs
Hidden categories : Articles needing additional references from June 2008 All articles needing additional references Articles with hAudio microformats
Singlechart usages for German Singlechart usages for Dutch100 Singlechart usages for UK Singlechart usages for Billboardhot100 Singlechart called without song
Talk Contents About Wikipedia 日本 語 Edit links This page was last edited on 9 April 2018 , at 00 : 34 . Text is available under the Creative Commons Attribution
- ShareAlike License ; additional terms may apply . By using this site , you agree to the Terms of Use and Privacy Policy . Wikipedia ® is a registered
trademark of the Wikimedia Foundation , Inc. , a non-profit organization . About Wikipedia
</item 4>

<item 5>
Joy to the World (Three Dog Night song)
Genre Rock Length 3 : 50 ( album ) 3 : 17 ( single ) Label Dunhill Songwriter ( s ) Hoyt Axton Producer ( s ) Richard Podolor Three Dog Night singles chronology
`` One Man Band '' ( 1970 ) `` Joy to the World '' ( 1971 ) `` Liar '' ( 1971 ) `` One Man Band '' ( 1970 ) `` Joy to the World '' ( 1971 ) `` Liar '' ( 1971 )
`` Joy to the World '' is a song written by Hoyt Axton , and made famous by the band Three Dog Night . The song is also popularly known by its opening lyric ,
`` Jeremiah was a bullfrog '' . Three Dog Night originally released the song on their fourth studio album , Naturally in November 1970 and subsequently released
an edited version of the song as a single in February 1971 .
</item 5>

<item 6>
Joy to the World (Three Dog Night song)
The single went on to sell 5 million copies worldwide , making it one of the best - selling singles of all time . In popular culture ( edit ) Singer -
songwriter Daniel Johnston likes this song , which inspired his own character Jeremiah the bullfrog , whom he often draws in his artwork and advertisements .
The frog is also featured on the cover of his album Hi , How Are You ( 1983 ) and has become an official music mascot for Johnston . It is sung by the son of
one of the main characters at the start of the film The Big Chill ( 1983 ) and is featured on the soundtrack . In The X-Files TV series ( Detour - S05E04 ) ,
Scully sings this song while she 's lost in a forest with Mulder to show him she is awake and he can rest . It is played during the end credits of the R - rated
animated film , Sausage Party ( 2016 ) , and later in the end credits of the PG - 13 - rated animated film A Very Spooky Night . It appears in 2017 and 2018 TV
commercials for Big Lots .
</item 6>

<item 7>
Joy to the World (Three Dog Night song)
Joy to the World ( Three Dog Night song ) - wikipedia Joy to the World ( Three Dog Night song ) Jump to : navigation , search This article needs additional
citations for verification . Please help improve this article by adding citations to reliable sources . Unsourced material may be challenged and removed . (
June 2008 ) ( Learn how and when to remove this template message ) `` Joy to the World '' 1971 French release Single by Three Dog Night from the album Naturally
B - side `` I Can Hear You Calling '' Released February 1971 ( 1971 - 02 ) Format 7 '' 45 RPM Recorded 1970 at American Recording Co .
</item 7>

<item 8>
Joy to the World (Three Dog Night song)
1965 -- 1975 ( 1993 ) 20th Century Masters -- The Millennium Collection : The Best of Three Dog Night ( 1999 ) Singles `` Nobody '' `` Try a Little Tenderness
'' `` One '' `` Easy to Be Hard '' `` Eli 's Coming '' `` Celebrate '' `` Mama Told Me ( Not to Come ) '' `` Out in the Country '' `` One Man Band '' `` Joy to
the World '' `` Liar '' `` An Old Fashioned Love Song '' `` Never Been to Spain '' `` The Family of Man '' `` Black & White '' `` Pieces of April '' `` Shambala
'' `` Let Me Serenade You '' `` The Show Must Go On '' `` Sure As I 'm Sittin ' Here '' `` Play Something Sweet ( Brickyard Blues ) '' `` Til the World Ends ''
`` Everybody Is a Masterpiece '' `` It 's a Jungle Out There '' Discography Book Billboard Year - End number one singles ( 1960 -- 1979 ) 1960 : `` Theme from A
Summer Place '' -- Percy Faith 1961 : `` Tossin ' and Turnin ' '' -- Bobby Lewis 1962 : ``
</item 8>

<item 9>
Joy to the World (Three Dog Night song)
ispot.tv . Retrieved December 24 , 2017 . External links ( edit ) Lyrics of this song at MetroLyrics Three Dog Night Danny Hutton Michael Allsup Paul Kingery
Pat Bautz Eddie Reasoner David Morgan Cory Wells Chuck Negron Jimmy Greenspoon Floyd Sneed Joe Schermie Skip Konte Al Ciner Gary Moon Richard Campbell Studio
albums Three Dog Night ( 1968 ) Suitable for Framing ( 1969 ) It Ai n't Easy ( 1970 ) Naturally ( 1970 ) Harmony ( 1971 ) Seven Separate Fools ( 1972 ) Cyan (
1973 ) Hard Labor ( 1974 ) Coming Down Your Way ( 1975 ) American Pastime ( 1976 ) It 's a Jungle ( 1983 ) Live albums Captured Live at the Forum ( 1969 )
Around the World with Three Dog Night ( 1973 ) Compilations Golden Bisquits ( 1971 ) Joy to the World : Their Greatest Hits ( 1975 ) The Best of 3 Dog Night (
1982 ) Celebrate : The Three Dog Night Story , 1965 -- 1975 (
</item 9>

<item 10>
Joy to the World (Three Dog Night song)
The song , which has been described by members of Three Dog Night as a `` kid 's song '' and a `` silly song '' , topped the main singles charts in North
America , was certified gold by the RIAA , and has since been covered by multiple artists . The song is featured prominently in the film The Big Chill . It is
sung by a child character at the beginning and the Three Dog Night recording is played over the end credits . It is also played at the end of every Denver
Broncos home victory . Notable playings of this song after Broncos victories included then - Chicago Bears head coach Abe Gibron 's singing along with the song
in 1973 ; and at the end of Super Bowl XXXII , played at Qualcomm Stadium in San Diego . It was also played at the end of Super Bowl XXXIII at Pro Player ( now
Hard Rock ) Stadium in Miami Gardens , Florida and Super Bowl 50 in Santa Clara , California . Contents ( hide ) 1 Background and recording 2 Charts and awards
3 In popular culture 4 References 5 External links Background and Recording ( edit ) Some of the words are nonsensical .
</item 10>
=== END ITEM id=af02fad03b61c994 ===

=== ITEM id=84a5776a0d3871d7 ===
Gold sentences and facts:
Sentence 0: Late in the 20th century , lithium became an important component of battery electrolytes and electrodes , because of its high electrode potential . Because of its low atomic mass , it has a high charge - and power - to - weight ratio . A typical lithium - ion battery can generate approximately 3 volts per cell , compared with 2.1 volts for lead - acid or 1.5 volts for zinc - carbon cells . Lithium - ion batteries , which are rechargeable and have a high energy density , should not be confused with lithium batteries , which are disposable ( primary ) batteries with lithium or its compounds as the anode . Other rechargeable batteries that use lithium include the lithium - ion polymer battery , lithium iron phosphate battery , and the nanowire battery .
  Fact 0.0: Late in the 20th century, lithium became an important component of battery electrolytes and electrodes.
  Fact 0.1: Lithium became an important battery component because of its high electrode potential.
  Fact 0.2: Lithium has a low atomic mass.
  Fact 0.3: Lithium has a high charge-to-weight ratio because of its low atomic mass.
  Fact 0.4: Lithium has a high power-to-weight ratio because of its low atomic mass.
  Fact 0.5: A typical lithium-ion battery can generate approximately 3 volts per cell.
  Fact 0.6: Lead-acid batteries are compared at 2.1 volts.
  Fact 0.7: Zinc-carbon cells are compared at 1.5 volts.
  Fact 0.8: Lithium-ion batteries are rechargeable.
  Fact 0.9: Lithium-ion batteries have a high energy density.
  Fact 0.10: Lithium-ion batteries should not be confused with lithium batteries.
  Fact 0.11: Lithium batteries are disposable (primary) batteries.
  Fact 0.12: Lithium batteries have lithium or its compounds as the anode.
  Fact 0.13: The lithium-ion polymer battery is a rechargeable battery that uses lithium.
  Fact 0.14: The lithium iron phosphate battery is a rechargeable battery that uses lithium.
  Fact 0.15: The nanowire battery is a rechargeable battery that uses lithium.
Sentence 1: Lithium and its compounds have several industrial applications , including heat - resistant glass and ceramics , lithium grease lubricants , flux additives for iron , steel and aluminium production , lithium batteries , and lithium - ion batteries . These uses consume more than three quarters of lithium production .
  Fact 1.0: Lithium and its compounds have several industrial applications.
  Fact 1.1: Lithium and its compounds are used in heat-resistant glass and ceramics.
  Fact 1.2: Lithium grease lubricants are an industrial application of lithium.
  Fact 1.3: Lithium and its compounds are used as flux additives for iron, steel and aluminium production.
  Fact 1.4: Lithium batteries are an industrial application of lithium.
  Fact 1.5: Lithium-ion batteries are an industrial application of lithium.
  Fact 1.6: Uses of lithium and its compounds consume more than three quarters of lithium production.

Retrieved items:
<item 1>
Lithium
Lithium - wikipedia Lithium Jump to : navigation , search This article is about the chemical element . For the use of lithium as a medication , see Lithium (
medication ) . For other uses , see Lithium ( disambiguation ) . It has been suggested that Lithium as an investment be merged into this article . ( Discuss )
Proposed since June 2017 . Lithium , Li Lithium floating in oil General properties Pronunciation / ˈlɪθiəm / ​ ( LI - thee - əm ) Appearance silvery - white
Standard atomic weight ( A ) ( 7000693800000000000 ♠ 6.938 , 7000699700000000000 ♠ 6.997 ) conventional : 7000694000000000000 ♠ 6.94 Lithium in the periodic
table Hydrogen Helium Lithium Beryllium Boron Carbon Nitrogen Oxygen Fluorine Neon Sodium Magnesium Aluminium Silicon Phosphorus Sulfur Chlorine Argon Potassium
Calcium Scandium Titanium Vanadium Chromium Manganese Iron Cobalt Nickel Copper Zinc Gallium Germanium Arsenic Selenium Bromine Krypton Rubidium
</item 1>

<item 2>
Lithium
When cut open , it exhibits a metallic luster , but moist air corrodes it quickly to a dull silvery gray , then black tarnish . It never occurs freely in nature
, but only in ( usually ionic ) compounds , such as pegmatitic minerals which were once the main source of lithium . Due to its solubility as an ion , it is
present in ocean water and is commonly obtained from brines . Lithium metal is isolated electrolytically from a mixture of lithium chloride and potassium
chloride . The nucleus of the lithium atom verges on instability , since the two stable lithium isotopes found in nature have among the lowest binding energies
per nucleon of all stable nuclides . Because of its relative nuclear instability , lithium is less common in the solar system than 25 of the first 32 chemical
elements even though its nuclei are very light : it is an exception to the trend that heavier nuclei are less common . For related reasons , lithium has
important uses in nuclear physics . The transmutation of lithium atoms to helium in 1932 was the first fully man - made nuclear reaction , and lithium deuteride
serves as a fusion fuel in staged thermonuclear weapons .
</item 2>

<item 3>
Lithium
Another significant mineral of lithium is lepidolite . A newer source for lithium is hectorite clay , the only active development of which is through the
Western Lithium Corporation in the United States . At 20 mg lithium per kg of Earth 's crust , lithium is the 25th most abundant element . According to the
Handbook of Lithium and Natural Calcium , `` Lithium is a comparatively rare element , although it is found in many rocks and some brines , but always in very
low concentrations . There are a fairly large number of both lithium mineral and brine deposits but only comparatively few of them are of actual or potential
commercial value . Many are very small , others are too low in grade . '' The US Geological Survey estimates that in 2010 , Chile had the largest reserves by
far ( 7.5 million tonnes ) and the highest annual production ( 8,800 tonnes ) . One of the largest reserve bases of lithium is in the Salar de Uyuni area of
Bolivia , which has 5.4 million tonnes . Other major suppliers include Australia , Argentina and China .
</item 3>

<item 4>
Lithium
Lithium hydroxide is a strong base and , when heated with a fat , produces a soap made of lithium stearate . Lithium soap has the ability to thicken oils , and
it is used to manufacture all - purpose , high - temperature lubricating greases . Metallurgy ( edit ) Lithium ( e.g. as lithium carbonate ) is used as an
additive to continuous casting mould flux slags where it increases fluidity , a use which accounts for 5 % of global lithium use ( 2011 ) . Lithium compounds
are also used as additives ( fluxes ) to foundry sand for iron casting to reduce veining . Lithium ( as lithium fluoride ) is used as an additive to aluminium
smelters ( Hall -- Héroult process ) , reducing melting temperature and increasing electrical resistance , a use which accounts for 3 % of production ( 2011 ) .
When used as a flux for welding or soldering , metallic lithium promotes the fusing of metals during the process and eliminates the forming of oxides by
absorbing impurities .
</item 4>

<item 5>
Lithium
Worldwide , this is one of the largest use for lithium compounds . Glazes containing lithium oxides are used for ovenware . Lithium carbonate ( Li CO ) is
generally used in this application because it converts to the oxide upon heating . Electrical and electronics ( edit ) Late in the 20th century , lithium became
an important component of battery electrolytes and electrodes , because of its high electrode potential . Because of its low atomic mass , it has a high charge
- and power - to - weight ratio . A typical lithium - ion battery can generate approximately 3 volts per cell , compared with 2.1 volts for lead - acid or 1.5
volts for zinc - carbon cells . Lithium - ion batteries , which are rechargeable and have a high energy density , should not be confused with lithium batteries
, which are disposable ( primary ) batteries with lithium or its compounds as the anode . Other rechargeable batteries that use lithium include the lithium -
ion polymer battery , lithium iron phosphate battery , and the nanowire battery . Lubricating greases ( edit ) Main article : Lithium grease The third most
common use of lithium is in greases .
</item 5>

<item 6>
Lithium
2 10 cm / mol ( 298 K ) Young 's modulus 4.9 GPa Shear modulus 4.2 GPa Bulk modulus 11 GPa Mohs hardness 0.6 Brinell hardness 5 MPa CAS Number 7439 - 93 - 2
History Discovery Johan August Arfwedson ( 1817 ) First isolation William Thomas Brande ( 1821 ) Main isotopes of lithium Isotope Abundance Half - life ( t )
Decay mode Product Li 5 % stable Li 95 % stable Li content may be as low as 3.75 % in natural samples . Li would therefore have a content of up to 96.25 % .
view talk references in Wikidata Lithium ( from Greek : λίθος , translit . lithos , lit . ' stone ' ) is a chemical element with symbol Li and atomic number 3 .
It is a soft , silvery - white alkali metal . Under standard conditions , it is the lightest metal and the lightest solid element . Like all alkali metals ,
lithium is highly reactive and flammable , and is stored in mineral oil .
</item 6>

<item 7>
Lithium
The stockpiled lithium was depleted in lithium - 6 by 75 % , which was enough to affect the measured atomic weight of lithium in many standardized chemicals ,
and even the atomic weight of lithium in some `` natural sources '' of lithium ion which had been `` contaminated '' by lithium salts discharged from isotope
separation facilities , which had found its way into ground water . Lithium was used to decrease the melting temperature of glass and to improve the melting
behavior of aluminium oxide when using the Hall - Héroult process . These two uses dominated the market until the middle of the 1990s . After the end of the
nuclear arms race , the demand for lithium decreased and the sale of Department of Energy stockpiles on the open market further reduced prices . In the mid
1990s , several companies started to extract lithium from brine which proved to be a less expensive than underground or open - pit mining . Most of the mines
closed or shifted their focus to other materials because only the ore from zoned pegmatites could be mined for a competitive price . For example , the US mines
near Kings Mountain , North Carolina closed before the beginning of the 21st century .
</item 7>

<item 8>
Lithium
Extraction ( edit ) Lithium salts are extracted from water in mineral springs , brine pools , and brine deposits . Lithium is present in seawater , but
commercially viable methods of extraction have yet to be developed . Another potential source of lithium is the leachates of geothermal wells , which are
carried to the surface . Recovery of lithium has been demonstrated in the field ; the lithium is separated by simple filtration . The process and environmental
costs are primarily those of the already - operating well ; net environmental impacts may thus be positive . Uses ( edit ) Estimates of global lithium uses in
2015 Ceramics and glass ( 32 % ) Batteries ( 35 % ) Lubricating greases ( 9 % ) Continuous casting ( 5 % ) Air treatment ( 5 % ) Polymers ( 4 % ) Primary
aluminum production ( 1 % ) Pharmaceuticals ( < 1 % ) Other ( 9 % ) Ceramics and glass ( edit ) Lithium oxide is widely used as a flux for processing silica ,
reducing the melting point and viscosity of the material and leading to glazes with improved physical properties including low coefficients of thermal expansion
.
</item 8>

<item 9>
Lithium
000,000 Portugal 200 60,000 N / A Russia - - 1,000,000 Serbia - - 1,000,000 United States 38,000 6,900,000 Zimbabwe 900 23,000 100,000 + World total 32,500
14,000,000 46,900,000 See also : Lithium minerals Although lithium is widely distributed on Earth , it does not naturally occur in elemental form due to its
high reactivity . The total lithium content of seawater is very large and is estimated as 230 billion tonnes , where the element exists at a relatively constant
concentration of 0.14 to 0.25 parts per million ( ppm ) , or 25 micromolar ; higher concentrations approaching 7 ppm are found near hydrothermal vents .
Estimates for the Earth 's crustal content range from 20 to 70 ppm by weight . In keeping with its name , lithium forms a minor part of igneous rocks , with the
largest concentrations in granites . Granitic pegmatites also provide the greatest abundance of lithium - containing minerals , with spodumene and petalite
being the most commercially viable sources .
</item 9>

<item 10>
Lithium
We are not aware of any discoveries of lithium , ' he said . '' Lithia ( `` lithium brine '' ) is associated with tin mining areas in Cornwall , England and an
evaluation project from 400 - metre deep test boreholes is under consideration . If successful the hot brines will also provide geothermal energy to power the
lithium extraction and refining process . Biological ( edit ) Lithium is found in trace amount in numerous plants , plankton , and invertebrates , at
concentrations of 69 to 5,760 parts per billion ( ppb ) . In vertebrates the concentration is slightly lower , and nearly all vertebrate tissue and body fluids
contain lithium ranging from 21 to 763 ppb . Marine organisms tend to bioaccumulate lithium more than terrestrial organisms . Whether lithium has a
physiological role in any of these organisms is unknown . History ( edit ) Johan August Arfwedson is credited with the discovery of lithium in 1817 Petalite (
LiAlSi O ) was discovered in 1800 by the Brazilian chemist and statesman José Bonifácio de Andrada e Silva in a mine on the island of Utö , Sweden .
</item 10>
=== END ITEM id=84a5776a0d3871d7 ===

=== ITEM id=15fc024c0232ff67 ===
Gold sentences and facts:
Sentence 0: The provisions relating to the right to property were changed a number of times . The Forty - Fourth Amendment of 1978 deleted the right to property from the list of fundamental rights A new provision , Article 300 - A , was added to the constitution , which provided that `` no person shall be deprived of his property save by authority of law '' . Thus if a legislature makes a law depriving a person of his property , there would be no obligation on the part of the State to pay anything as compensation . The aggrieved person shall have no right to move the court under Article 32 . Thus , the right to property is no longer a fundamental right , though it is still a constitutional right . If the government appears to have acted unfairly , the action can be challenged in a court of law by aggrieved citizens .
  Fact 0.0: The provisions relating to the right to property were changed a number of times.
  Fact 0.1: The Forty-Fourth Amendment of 1978 deleted the right to property from the list of fundamental rights.
  Fact 0.2: A new provision, Article 300-A, was added to the constitution.
  Fact 0.3: Article 300-A provides that no person shall be deprived of his property save by authority of law.
  Fact 0.4: If a legislature makes a law depriving a person of his property, the State has no obligation to pay anything as compensation.
  Fact 0.5: An aggrieved person shall have no right to move the court under Article 32.
  Fact 0.6: The right to property is no longer a fundamental right.
  Fact 0.7: The right to property is still a constitutional right.
  Fact 0.8: If the government appears to have acted unfairly, aggrieved citizens can challenge the action in a court of law.

Retrieved items:
<item 1>
Fundamental rights in India
Jump up ^ Sovereign means possessing supreme or ultimate power . `` in modern democracies the people 's will is in theory sovereign '' synonyms : supreme ,
absolute , unlimited , unrestricted , unrestrained , unbounded , boundless , infinite , ultimate , total , unconditional , full , utter , paramount ; principal
, chief , dominant , predominant Jump up ^ `` What causes judicial delay ? Judgements diluting time frames in Code of Civil Procedure worsen the problem of
adjournments '' . Retrieved 13 September 2017 . Jump up ^ `` Paras 316 and 317 of Kesavananda Bharati v. State of Kerala , ( AIR 1973 SC 1461 ) '' . 1973 .
Retrieved 21 October 2016 . Jump up ^ `` IX Schedule laws open to review '' . 2007 . Retrieved 21 October 2017 . Jump up ^ 44th Amendment Act , 1978 . Jump up ^
Mahapatra , Dhananjay ( 28 February 2009 ) . `` Should right to property return ? '' . The Times of India . Retrieved 8 July 2010 .
</item 1>

<item 2>
Fundamental rights in India
The provisions relating to the right to property were changed a number of times . The Forty - Fourth Amendment of 1978 deleted the right to property from the
list of fundamental rights A new provision , Article 300 - A , was added to the constitution , which provided that `` no person shall be deprived of his
property save by authority of law '' . Thus if a legislature makes a law depriving a person of his property , there would be no obligation on the part of the
State to pay anything as compensation . The aggrieved person shall have no right to move the court under Article 32 . Thus , the right to property is no longer
a fundamental right , though it is still a constitutional right . If the government appears to have acted unfairly , the action can be challenged in a court of
law by aggrieved citizens . The liberalisation of the economy and the government 's initiative to set up special economic zones has led to many protests by
farmers and have led to calls for the reinstatement of the fundamental right to private property .
</item 2>

<item 3>
Fundamental rights in India
It is the principal foundation of all other rights and liberties , and guarantees the following : Equality before law : Article 14 of the constitution
guarantees that all people shall be equally protected by the laws of the country . It means that the State will treat people in the same circumstances alike .
This article also means that individuals , whether citizens of India or otherwise shall be treated differently if the circumstances are different . Social
equality and equal access to public areas : Article 15 of the constitution states that no person shall be discriminated on the basis of religion , race , caste
, sex or place of birth . Every person shall have equal access to public places like public parks , museums , wells , bathing ghats and temples etc . However ,
the State may make any special provision for women and children . Special provisions may be made for the advancements of any socially or educationally backward
class or scheduled castes or scheduled tribes . Equality in matters of public employment : Article 16 of the constitution lays down that the State can not
discriminate against anyone in the matters of employment . All citizens can apply for government jobs . There are some exceptions .
</item 3>

<item 4>
Fundamental rights in India
In 2007 , Supreme Court ruled that there could not be any blanket immunity from judicial review for the laws inserted in the Ninth Schedule . Apex court also
stated it shall examine laws included in the Ninth Schedule after 1973 for any incompatibility with the basic structure doctrine . Amendment to Article 31C (
edit ) Section 4 of the 42nd Amendment , had changed Article 31C of the constitution to accord precedence to the Directive Principles ( earlier applicable only
to clauses b & c of Article 39 ) over the fundamental rights of individuals . In Minerva Mills v. Union of India case , supreme court ruled that the amendment
to the Article 31C is not valid and ultra vires . Right to property ( edit ) The Constitution originally provided for the right to property under Articles 19
and 31 . Article 19 guaranteed to all citizens the right to acquire , hold and dispose of property . Article 31 provided that `` no person shall be deprived of
his property save by authority of law . '' It also provided that compensation would be paid to a person whose property has been taken for public purposes .
</item 4>

<item 5>
Fundamental rights in India
The six fundamental rights recognised by the Indian constitution are the right to equality , right to freedom , right against exploitation , right to freedom of
religion , cultural and educational rights , right to constitutional remedies and right to privacy . The right to equality includes equality before law ,
prohibition of discrimination on grounds of religion , race , caste , gender or place of birth , and equality of opportunity in matters of employment ,
abolition of untouchability and abolition of titles . The right to freedom includes freedom of speech and expression , assembly , association or union or
cooperatives , movement , residence , and right to practice any profession or occupation , right to life and liberty , protection in respect to conviction in
offences and protection against arrest and detention in certain cases . The right against exploitation prohibits all forms of forced labour , child labour and
trafficking of human beings . The right to freedom of religion includes freedom of conscience and free profession , practice , and propagation of religion ,
freedom to manage religious affairs , freedom from certain taxes and freedom from religious instructions in certain educational institutes . Cultural and
educational rights preserve the right of any section of citizens to conserve their culture , language or script , and right of minorities to establish and
administer educational institutions of their choice .
</item 5>

<item 6>
Fundamental rights in India
. The Times of India . Retrieved 8 July 2010 . Jump up ^ `` Court rejects plea to make property a fundamental right '' . The Hindu . Chennai , India . 19
October 2010 . Jump up ^ `` Education is now a fundamental right of every child '' . Hindustan Times . 1 April 2010 . Archived from the original on 9 July 2010
. Retrieved 8 July 2010 . Footnotes ( edit ) Links to related articles ( hide ) Government of India Constitution Constitution Fundamental Rights , Directive
Principles and Fundamental Duties Federalism ( Union List State List Concurrent List ) Executive Union Government Prime Minister ( list ) Cabinet Ministers
President ( list ) Vice President ( list ) Civil Services All India Services ( IAS IFS / IFoS IPS ) Cabinet Secretary Secretaries ( Defence Finance Foreign Home
) State governments Chief Minister ( list ) Chief Secretary Governors ( list ) Legislative Parliament Lok Sabha ( Speaker ) Rajya Sabha ( Chairman ) Vidhan
Sabha Vidhan Parishad Leader of the Opposition Judiciary Supreme Court ( Chief Justice ) High Courts District Courts Retrieved from ``
https://en.wikipedia.org/w/index.
</item 6>

<item 7>
Fundamental rights in India
They are covered in Part III ( Articles 12 to 35 ) of Indian constitution . Contents ( hide ) 1 Genesis 2 Significance and characteristics 3 Right to equality 4
Right to freedom 5 Right against exploitation 6 Right to freedom of religion 7 Right to life 8 Cultural and Educational rights 9 Right to constitutional
remedies 10 Right to Privacy 11 Critical analysis 12 Amendments 13 Fundamental rights not sacrosanct 13.1 Validity of Article 31B 13.2 Amendment to Article 31C
13.3 Right to property 13.4 Right to education 14 See also 15 References 16 Footnotes Genesis ( edit ) The development of constitutionally guaranteed
fundamental human rights in India was inspired by historical examples such as England 's Bill of Rights ( 1689 ) , the United States Bill of Rights ( approved
on 17 September 1787 , final ratification on 15 December 1791 ) and France 's Declaration of the Rights of Man ( created during the revolution of 1789 , and
ratified on 26 August 1789 ) .
</item 7>

<item 8>
Fundamental rights in India
The right to constitutional remedies is present for enforcement of Fundamental Rights . The right to privacy is an intrinsic part of Article 21 that protects
life and liberty of the citizens . Fundamental rights for Indians have also been aimed at overturning the inequalities of pre-independence social practices .
Specifically , they have also been used to abolish untouchability and thus prohibit discrimination on the grounds of religion , race , caste , sex , or place of
birth . They also forbid trafficking of human beings and forced labour ( a crime ) . They also protect cultural and educational rights of religious and
linguistic minorities by allowing them to preserve their languages and also establish and administer their own education institutions . They are covered in Part
III ( Articles 12 to 35 ) of Indian constitution .
</item 8>

<item 9>
Fundamental rights in India
These fundamental rights help not only in protection but also the prevention of gross violations of human rights . They emphasise on the fundamental unity of
India by guaranteeing to all citizens the access and use of the same facilities , irrespective of background . Some fundamental rights apply for persons of any
nationality whereas others are available only to the citizens of India . The right to life and personal liberty is available to all people and so is the right
to freedom of religion . On the other hand , freedoms of speech and expression and freedom to reside and settle in any part of the country are reserved to
citizens alone , including non-resident Indian citizens . The right to equality in matters of public employment can not be conferred to overseas citizens of
India . Fundamental rights primarily protect individuals from any arbitrary state actions , but some rights are enforceable against individuals . For instance ,
the Constitution abolishes untouchability and also prohibits begar . These provisions act as a check both on state action as well as the action of private
individuals . However , these rights are not absolute or uncontrolled and are subject to reasonable restrictions as necessary for the protection of general
welfare . They can also be selectively curtailed .
</item 9>

<item 10>
Fundamental rights in India
72 ∙ 73 ∙ 74 ∙ 75 ∙ 76 ∙ 77 ∙ 78 ∙ 79 ∙ 80 ∙ 81 ∙ 82 ∙ 83 ∙ 84 ∙ 85 ∙ 86 ∙ 87 ∙ 88 ∙ 89 ∙ 90 ∙ 91 ∙ 92 ∙ 93 ∙ 94 ∙ 95 ∙ 96 ∙ 97 ∙ 98 ∙ 99 ∙ 100 ∙ 101 Related
topics ( show ) Union List State List Concurrent List Basic structure doctrine Uniform civil code Fundamental Rights are the basic rights of the people and the
charter of rights contained in Part III ( Article 12 to 35 ) of Constitution of India . It guarantees civil liberties such that all Indians can lead their lives
in peace and harmony as citizens of India .
</item 10>
=== END ITEM id=15fc024c0232ff67 ===

=== ITEM id=7a452ffb0c10a249 ===
Gold sentences and facts:
Sentence 0: Donald Trump is the 45th and current president , inaugurated on January 20 , 2017 .
  Fact 0.0: Donald Trump is the 45th president.
  Fact 0.1: Donald Trump is the current president.
  Fact 0.2: Donald Trump was inaugurated on January 20, 2017.
Sentence 1: President of the United States of America Presidential Seal Presidential Standard Incumbent Donald Trump since January 20 , 2017 Executive Branch of the U.S. Government Executive Office of the President Style Mr. President ( informal ) The Honorable ( formal ) His Excellency ( international correspondence ) Status Head of State Head of Government Member of Cabinet Domestic Policy Council National Economic Council National Security Council Residence White House Seat Washington , D.C. Appointer Electoral College of the United States Term length Four years , renewable once ( Amendment XXII ) Constituting instrument United States Constitution Inaugural holder George Washington Formation March 4 , 1789 ( 229 years ago ) ( 1789 - 03 - 04 ) Salary $400,000 annually Website whitehouse.gov
  Fact 1.0: The office title is President of the United States of America.
  Fact 1.1: The office has a Presidential Seal and a Presidential Standard.
  Fact 1.2: Donald Trump has been the incumbent since January 20, 2017.
  Fact 1.3: The office is in the Executive Branch of the U.S. Government.
  Fact 1.4: The office is in the Executive Office of the President.
  Fact 1.5: The informal style is Mr. President.
  Fact 1.6: The formal style is The Honorable.
  Fact 1.7: The style for international correspondence is His Excellency.
  Fact 1.8: The status is Head of State and Head of Government.
  Fact 1.9: The president is a member of the Cabinet.
  Fact 1.10: The president is a member of the Domestic Policy Council.
  Fact 1.11: The president is a member of the National Economic Council.
  Fact 1.12: The president is a member of the National Security Council.
  Fact 1.13: The residence is the White House.
  Fact 1.14: The seat is in Washington, D.C.
  Fact 1.15: The appointer is the Electoral College of the United States.
  Fact 1.16: The term length is four years, renewable once (Amendment XXII).
  Fact 1.17: The constituting instrument is the United States Constitution.
  Fact 1.18: The inaugural holder was George Washington.
  Fact 1.19: The office was formed on March 4, 1789, listed as 229 years ago (1789-03-04).
  Fact 1.20: The salary is $400,000 annually.
  Fact 1.21: The website is whitehouse.gov.

Retrieved items:
<item 1>
President of the United States
Book Category Portal Heads of state and government of North America Heads of state Sovereign states Antigua and Barbuda Bahamas Barbados Belize Canada Costa
Rica Cuba Dominica Dominican Republic El Salvador Grenada Guatemala Haiti Honduras Jamaica Mexico Nicaragua Panama Saint Kitts and Nevis Saint Lucia Saint
Vincent and the Grenadines Trinidad and Tobago United States Heads of government Sovereign states Antigua and Barbuda Bahamas Barbados Belize Canada Costa Rica
Cuba Dominica Dominican Republic El Salvador Grenada Guatemala Haiti Honduras Jamaica Mexico Nicaragua Panama Saint Kitts and Nevis Saint Lucia Saint Vincent
and the Grenadines Trinidad and Tobago United States VIAF : 123879793 LCCN : n80001199 ISNI : 0000 0001 2106 1812 GND : 2033432 - 1 SUDOC : 026377519 BNF :
cb11863688j ( data ) Retrieved from `` https://en.wikipedia.org/w/index.php?title=President_of_the_United_States&oldid=837764569 '' Categories : Presidents of
the United States Presidency of the United States United States presidential history 1789 establishments in the United States Hidden categories : CS1 : Julian
-- Gregorian uncertainty
</item 1>

<item 2>
President of the United States
Kennedy ( 1961 -- 1963 ) Lyndon B. Johnson ( 1963 -- 1969 ) Richard Nixon ( 1969 -- 1974 ) Gerald Ford ( 1974 -- 1977 ) Jimmy Carter ( 1977 -- 1981 ) Ronald
Reagan ( 1981 -- 1989 ) George H.W. Bush ( 1989 -- 1993 ) Bill Clinton ( 1993 -- 2001 ) George W. Bush ( 2001 -- 2009 ) Barack Obama ( 2009 -- 2017 ) Donald
Trump ( 2017 -- present ) Presidency timelines Wilson Harding Coolidge Hoover F.D. Roosevelt Truman Eisenhower Kennedy L.B. Johnson Nixon Ford Carter Reagan
G.H.W. Bush Clinton G.W. Bush Obama Trump Book Category Chief executives of the United States Federal President of the United States State governors ( current
list ) Alabama Alaska Arizona Arkansas California list Colorado Connecticut Delaware Florida Georgia Hawaii list Idaho Illinois list Indiana list Iowa list
Kansas Kentucky list Louisiana Maine list Maryland list Massachusetts Michigan list Minnesota list Mississippi Missouri Montana Nebraska list Nevada New
Hampshire list New Jersey list New Mexico list New York list North Carolina list North Dakota list Ohio Oklahoma list Oregon
</item 2>

<item 3>
President of the United States
The President of the United States ( POTUS / ˈpoʊtəs / POH - təs ) is the head of state and head of government of the United States of America . The president
directs the executive branch of the federal government and is the commander - in - chief of the United States Armed Forces . In contemporary times , the
president is looked upon as the world 's most powerful political figure and as the leader of the only current global superpower . The role includes
responsibility for the world 's most expensive military that has the second largest nuclear arsenal . The president also leads the nation with the largest
economy by nominal GDP . The president possesses significant domestic and international hard and soft power . Article II of the Constitution establishes the
executive branch of the federal government . It vests the executive power of the United States in the president . The power includes the execution and
enforcement of federal law , alongside the responsibility of appointing federal executive , diplomatic , regulatory and judicial officers , and concluding
treaties with foreign powers with the advice and consent of the Senate .
</item 3>

<item 4>
President of the United States
Images of documents written by U.S. presidents Presidents of the United States ( list ) George Washington ( 1789 -- 1797 ) John Adams ( 1797 -- 1801 ) Thomas
Jefferson ( 1801 -- 1809 ) James Madison ( 1809 -- 1817 ) James Monroe ( 1817 -- 1825 ) John Quincy Adams ( 1825 -- 1829 ) Andrew Jackson ( 1829 -- 1837 )
Martin Van Buren ( 1837 -- 1841 ) William Henry Harrison ( 1841 ) John Tyler ( 1841 -- 1845 ) James K. Polk ( 1845 -- 1849 ) Zachary Taylor ( 1849 -- 1850 )
Millard Fillmore ( 1850 -- 1853 ) Franklin Pierce ( 1853 -- 1857 ) James Buchanan ( 1857 -- 1861 ) Abraham Lincoln ( 1861 -- 1865 ) Andrew Johnson ( 1865 --
1869 ) Ulysses S. Grant ( 1869 -- 1877 ) Rutherford B. Hayes ( 1877 -- 1881 ) James A.
</item 4>

<item 5>
President of the United States
President of the United States - wikipedia President of the United States Jump to : navigation , search `` POTUS '' redirects here . For the political talk
radio channel , see P.O.T.U.S. ( Sirius XM ) . For other uses , see President of the United States ( disambiguation ) . For a list , see List of Presidents of
the United States . President of the United States of America Presidential Seal Presidential Standard Incumbent Donald Trump since January 20 , 2017 Executive
Branch of the U.S. Government Executive Office of the President Style Mr. President ( informal ) The Honorable ( formal ) His Excellency ( international
correspondence ) Status Head of State Head of Government Member of Cabinet Domestic Policy Council National Economic Council National Security Council Residence
White House Seat Washington , D.C. Appointer Electoral College of the United States Term length Four years , renewable once ( Amendment XXII ) Constituting
instrument United States Constitution Inaugural holder George Washington Formation March 4 , 1789 ( 229 years ago ) ( 1789 - 03 - 04 ) Salary $400,000 annually
Website whitehouse.gov This article is part of a series on the Politics of the United
</item 5>

<item 6>
President of the United States
The president is further empowered to grant federal pardons and reprieves , and to convene and adjourn either or both houses of Congress under extraordinary
circumstances . The president directs the foreign and domestic policies of the United States , and takes an active role in promoting his policy priorities to
members of Congress . In addition , as part of the system of checks and balances , Article One of the United States Constitution gives the president the power
to sign or veto federal legislation . Since the office of president was established in 1789 , its power has grown substantially , as has the power of the
federal government as a whole . Through the Electoral College , the registered voters indirectly elect the president and vice president to a four - year term .
This is the only federal election in the United States which is not decided by popular vote . Nine vice presidents became president by virtue of a president 's
intra-term death or resignation . The Twenty - second Amendment precludes any United States citizen from being elected president for a third term .
</item 6>

<item 7>
President of the United States
000 annually Website whitehouse.gov This article is part of a series on the Politics of the United States of America Federal Government ( show ) Constitution of
the United States Law Taxation Legislature ( show ) United States Congress House of Representatives Speaker Paul Ryan ( R ) Majority Leader Kevin McCarthy ( R )
Minority Leader Nancy Pelosi ( D ) Congressional districts United States Senate President Mike Pence ( R ) President Pro Tempore Orrin Hatch ( R ) President Pro
Tempore Emeritus Patrick Leahy ( D ) Majority Leader Mitch McConnell ( R ) Minority Leader Chuck Schumer ( D ) Executive ( show ) President of the United States
Donald Trump ( R ) Vice President of the United States Mike Pence ( R ) Cabinet Federal agencies Executive Office Judiciary ( show ) Supreme Court of the United
States Chief Justice John Roberts Kennedy Thomas Ginsburg Breyer Alito Sotomayor Kagan Gorsuch Courts of Appeals District Courts ( list ) Other tribunals
Elections ( show ) Presidential elections Midterm elections Off - year elections Political parties ( show ) Democratic Republican Third parties Federalism (
show ) State Government Governors Legislatures ( List ) State courts Local government Other countries Atlas The President of the United States ( POTUS / ˈpoʊtəs
/
</item 7>

<item 8>
President of the United States
Świątczak , Wasilewska , Iwona . `` The Toughest Season in the White House '' : The Rhetorical Presidency and the State of the Union Address , 1953 -- 1992 .
PhD thesis . University of Helsinki , 2014 . ISBN 978 - 951 - 51 - 0248 - 5 . Primary sources Waldman , Michael -- Stephanopoulos , George . My Fellow Americans
: The Most Important Speeches of America 's presidents , from George Washington to George W. Bush . Sourcebooks Trade . 2003 . ISBN 1 - 4022 - 0027 - 7 Jacobs ,
Ron . Interview with Joseph G. Peschek and William Grover , authors of The Unsustainable Presidency , a book offering an analysis of the role the US President
plays in economics and politics External links Wikisource has original text related to this article : President of the United States Wikimedia Commons has media
related to Presidents of the United States . Wikiquote has quotations related to : President of the United States Official `` Executive Office of the President
'' . Retrieved January 21 , 2009 .
</item 8>

<item 9>
President of the United States
Jump up ^ The informal term POTUS originated in the Phillips Code , a shorthand method created in 1879 by Walter P. Phillips for the rapid transmission of press
reports by telegraph . Jump up ^ The nine vice presidents who succeeded to the presidency upon their predecessor 's death or resignation and finished - out that
unexpired term are John Tyler ( 1841 ) ; Millard Fillmore ( 1850 ) ; Andrew Johnson ( 1865 ) ; Chester A. Arthur ( 1881 ) ; Theodore Roosevelt ( 1901 ) ; Calvin
Coolidge ( 1923 ) ; Harry S. Truman ( 1945 ) ; Lyndon B. Johnson ( 1963 ) ; and Gerald Ford ( 1974 ) . Jump up ^ Foreign - born American citizens who met the
age and residency requirements at the time the Constitution was adopted were also eligible for the presidency . However , this allowance has since become
obsolete . References Jump up ^ `` How To Address The President ; He Is Not Your Excellency Or Your Honor , But Mr. President '' . The New York Times . August 2
, 1891 .
</item 9>

<item 10>
President of the United States
Administrative powers Suffice it to say that the President is made the sole repository of the executive powers of the United States , and the powers entrusted
to him as well as the duties imposed upon him are awesome indeed . William Rehnquist , Nixon v. General Services Administration , 433 U.S. 425 ( 1977 ) (
dissenting opinion ) The president is the head of the executive branch of the federal government and is constitutionally obligated to `` take care that the laws
be faithfully executed '' . The executive branch has over four million employees , including members of the military . Presidents make numerous executive branch
appointments : an incoming president may make up to 6,000 before taking office and 8,000 more while serving . Ambassadors , members of the Cabinet , and other
federal officers , are all appointed by a president with the `` advice and consent '' of a majority of the Senate . When the Senate is in recess for at least
ten days , the president may make recess appointments . Recess appointments are temporary and expire at the end of the next session of the Senate . The power of
a president to fire executive officials has long been a contentious political issue .
</item 10>
=== END ITEM id=7a452ffb0c10a249 ===
