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

=== ITEM id=e4f857fde1e9a8b4 ===
Gold sentences and facts:
Sentence 0: The Eagles and the Patriots met again in Super Bowl LII , following the 2017 season , with the Eagles taking their revenge 41 -- 33 .
  Fact 0.0: The Eagles and the Patriots met again in Super Bowl LII.
  Fact 0.1: Super Bowl LII followed the 2017 season.
  Fact 0.2: The Eagles took their revenge in Super Bowl LII.
  Fact 0.3: The Eagles won Super Bowl LII 41 to 33.

Retrieved items:
<item 1>
Super Bowl XXXIX
For the Eagles , with the loss , the city of Philadelphia would not see its first championship since the 76ers swept the 1983 NBA Finals until the next time
their city 's teams played for a championship , when the Phillies won the 2008 World Series . The Eagles did not return to the Super Bowl until 2017 , also with
a 13 - 3 record , and defeating the Falcons and Vikings in the playoffs . The Eagles and the Patriots met again in Super Bowl LII , following the 2017 season ,
with the Eagles taking their revenge 41 -- 33 . This is the last time that a team has won back - to - back Super Bowls , with the Patriots losing their second
playoff game a year later and then , for the next eight years , every team either losing their first playoff game or missing them altogether . The streak was
broken in 2014 when the Seahawks defeated the Panthers and ultimately advanced to the Super Bowl , only to lose to a Patriots goal line stand .
</item 1>

<item 2>
Super Bowl XXXIX
The next year , the Patriots , having faltered down the stretch , entered those playoffs as the second seed behind the Broncos , who then defeated New England
in the AFC Title Game that year en route to a title of their own in Super Bowl 50 . Two years later , the Patriots , defending champions themselves from Super
Bowl LI , returned to repeat their title only to lose the aforementioned Super Bowl LII . Final statistics ( edit ) Sources : NFL.com Super Bowl XXXIX , Super
Bowl XXXIX Play Finder NE , Super Bowl XXXIX Play Finder Phi Statistical comparison ( edit ) New England Patriots Philadelphia Eagles First downs 21 24 First
downs rushing 6 First downs passing 14 18 First downs penalty Third down efficiency 4 / 12 9 / 16 Fourth down efficiency 0 / 0 0 / 0 Net yards rushing 112 45
Rushing attempts 28 17 Yards per rush 4.0 2.
</item 2>

<item 3>
Super Bowl XXXIX
Tom Brady took a knee to run out the clock , clinching the 3rd Super Bowl title in 4 years for the Patriots , and in the eyes of many establishing themselves as
a dynasty . Adam Vinatieri warms up before the game An Eagles offensive play Philadelphia kicks off after scoring a touchdown Brian Westbrook 's touchdown catch
( photo 1 of 2 ) Brian Westbrook 's touchdown catch ( photo 2 of 2 ) Box score ( edit ) Super Bowl XXXIX : New England Patriots 24 , Philadelphia Eagles 21
Total Patriots ( AFC ) 0 7 7 10 24 Eagles ( NFC ) 0 7 7 7 21 at Alltel Stadium , Jacksonville , Florida Date : February 6 , 2005 Game time : 6 : 38 p.m. EST
Game weather : 59 ° F ( 15 ° C ) , clear ( hide ) Scoring summary Quarter Time Drive Team Scoring information Score Plays Yards TOP NE PHI 9 : 55 9 81 4 : 36
PHI L.J.
</item 3>

<item 4>
Super Bowl XXXIX
Coast Guard Academy , and U.S. Army Herald Trumpets . Coin toss Youth football players from Jacksonville : Tyler Callahan , Tyler Deal , Lawrence McCauley , and
Jacob Santana ; New Orleans NFL Junior Player Development coach Tamaris Jackson Halftime show Paul McCartney TV in the United States Network Fox Announcers Joe
Buck , Troy Aikman , Cris Collinsworth , Pam Oliver and Chris Myers Nielsen ratings 41.1 ( est. 86 million viewers ) Market share 62 Cost of 30 - second
commercial $2.3 million ← XXXVIII Super Bowl XL → Super Bowl XXXIX was an American football game played between the American Football Conference ( AFC )
champion New England Patriots and the National Football Conference ( NFC ) champion Philadelphia Eagles to decide the National Football League ( NFL ) champion
for the 2004 season . The Patriots defeated the Eagles by the score of 24 -- 21 . The game was played on February 6 , 2005 , at Alltel Stadium in Jacksonville ,
Florida , the first time the Super Bowl was played in that city .
</item 4>

<item 5>
Super Bowl XXXIX
The Patriots , who entered the Super Bowl after compiling a 14 -- 2 regular season record , became the first team since the 1997 -- 1998 Denver Broncos to win
consecutive Super Bowls . New England also became the second team after the Dallas Cowboys to win three Super Bowls in four years . The Eagles were making their
second Super Bowl appearance after posting a 13 -- 3 regular season record . The game was close throughout , with the teams battling to a 14 -- 14 tie by the
end of the third quarter . The Patriots then scored 10 points in the 4th quarter with Corey Dillon 's 2 - yard touchdown run and Adam Vinatieri 's 22 - yard
field goal . The Eagles then cut their deficit to 24 -- 21 , with quarterback Donovan McNabb 's 30 - yard touchdown pass to receiver Greg Lewis , with 1 : 48
remaining in the game but could not sustain the comeback . Overall , New England forced four turnovers , while Patriots wide receiver Deion Branch was named
Super Bowl MVP for recording 133 receiving yards and tied the Super Bowl record with 11 catches .
</item 5>

<item 6>
Super Bowl XXXIX
Aftermath ( edit ) The Patriots ' Super Bowl win was the third championship for Boston - area sports teams in 12 months , following the Patriots winning Super
Bowl XXXVIII the year before and the Red Sox winning the World Series -- first in 86 years -- three months earlier . This marked the first time since 1989 --
1990 in the San Francisco Bay Area that the same market has had 2 Super Bowl and World Series winners in 12 months . The Patriots would later appear in Super
Bowl XLII and Super Bowl XLVI , losing both to Eli Manning 's New York Giants , before winning Super Bowl XLIX against the Seattle Seahawks , 28 -- 24 , ten
years later and Super Bowl LI against the Atlanta Falcons , 34 - 28 , two years later . The latest Super Bowl win for the Patriots makes it ten titles among the
four Boston teams ( 5 by the Patriots , 3 by the Red Sox , and one each by the Celtics and Bruins ) since 2002 .
</item 6>

<item 7>
Philadelphia Eagles
Foles had his best game since week 15 and threw for 352 passing yards and 3 touchdowns including an impressive flea flicker touchdown pass . This win marked the
franchise 's third Super Bowl appearance and a berth in Super Bowl LII for a 2005 rematch against Tom Brady and the New England Patriots . With Foles at the
helm , the Eagles started off the game strong , leading the Patriots 22 -- 12 at halftime . In the second half the Patriots gained momentum and for the first
time took the lead in the 4th quarter 33 -- 32 . The Eagles rallied back and scored an 11 - yard Touchdown to tight end Zach Ertz . Later the Eagles converted
on a 46 - yard field goal by Jake Elliott to lead 41 -- 33 . On the final play of the game with 9 seconds left , Brady threw a hail - mary pass directed toward
tight end Rob Gronkowski which was bobbled by Patrick Robinson before being dropped in the end - zone as time expired , securing the win for the Eagles 41 -- 33
, and ending the 58 - year long championship drought .
</item 7>

<item 8>
Super Bowl XXXIX
Super Bowl XXXIX - wikipedia Super Bowl XXXIX Jump to : navigation , search Super Bowl XXXIX New England Patriots ( AFC ) Philadelphia Eagles ( NFC ) 24 21
Total NE 0 7 7 10 24 PHI 0 7 7 7 21 Date February 6 , 2005 ( 2005 - 02 - 06 ) Stadium Alltel Stadium , Jacksonville , Florida MVP Deion Branch , Wide Receiver
Favorite Patriots by 7 Referee Terry McAulay Attendance 78,125 Future Hall of Famers Patriots : None Eagles : Brian Dawkins , Terrell Owens Ceremonies National
anthem The combined choirs of the U.S. Military Academy , the U.S. Naval Academy , the U.S. Air Force Academy , and the U.S. Coast Guard Academy , and U.S. Army
Herald Trumpets .
</item 8>

<item 9>
Super Bowl XXXIX
It was the first time New England trailed during the entire postseason . On their ensuing drive , the Patriots moved the ball to the Eagles four - yard line ,
mainly on plays by running back Corey Dillon , who caught two screen passes for 29 yards and rushed for 25 . But quarterback Tom Brady fumbled the ball on a
fake handoff Play - action pass and Philadelphia defender Darwin Walker recovered it . However , the Eagles could not take advantage of the turnover and had to
punt after 3 plays . Eagles punter Dirk Johnson 's punt went just 29 yards , giving the Patriots the ball at Philadelphia 's 37 - yard line . The Patriots then
drove 37 yards to score on Brady 's 4 - yard pass to receiver David Givens with 1 : 10 remaining in the period , tying the game 7 -- 7 by halftime . It was only
the second halftime tie in Super Bowl history and the first time both of the game 's first two quarters ended tied .
</item 9>

<item 10>
Super Bowl XXXIX
The 19 yard gain , and a roughing - the - passer penalty on Philadelphia defensive lineman Corey Simon on the same play , set up kicker Adam Vinatieri 's 22 -
yard field goal with 8 : 43 left in the game to increase the score 24 -- 14 in favor of New England . In all three Patriots ' Super Bowl wins in the decade ,
they held a double digit lead in the 4th quarter . The Eagles responded with a long McNabb completion to Owens . However , after reaching the New England 36 -
yard line , McNabb fired a pass over the head of Dorsey Levens , where Tedy Bruschi was waiting to intercept it at the Patriots 24 - yard line . At this point ,
there was only 7 : 20 to play in the game , with the Patriots still up by 10 points . The Eagles did force New England to punt , and got the ball back at their
own 21 - yard line , but with 5 : 40 left in the game . The Eagles then drove 79 yards in 13 plays , however they used a full huddle for the entire drive .
</item 10>
=== END ITEM id=e4f857fde1e9a8b4 ===

=== ITEM id=5a6386d78cfa5bb4 ===
Gold sentences and facts:
Sentence 0: Whig ( 17 ) Tory ( 11 ) Conservative ( 17 ) Peelite ( 1 ) Liberal ( 7 ) Labour ( 6 ) National Labour ( 1 ) Portrait Name ( Birth -- Death ) Term of office Mandate Ministerial portfolios held as Prime Minister Party Ministry Monarch ( Reign ) Duration The Right Honourable Robert Walpole 1st Earl of Orford KG KB PC MP for King 's Lynn ( 1676 -- 1745 ) 4 April 1721 11 February 1742 1722 First Lord of the Treasury Chancellor of the Exchequer Leader of the House of Commons Whig Walpole -- Townshend George I ( 1714 -- 1727 ) 1727 George II ( 1727 -- 1760 ) 1734 Walpole 1741 7003761900000000000 ♠ 20 years and 314 days The Right Honourable Spencer Compton 1st Earl of Wilmington KG KB PC ( 1673 -- 1743 ) 16 February 1742 2 July 1743 -- First Lord of the Treasury Whig Carteret 7002502000000000000 ♠ 1 year and 137 days ​ The Right Honourable Henry Pelham FRS MP for Sussex ( 1694 -- 1754 ) 27 August 1743 6 March 1754 1747 First Lord of the Treasury Chancellor of the Exchequer Leader of the House of Commons Whig Broad Bottom I Broad Bottom II 7003384500000000000 ♠ 10 years and 192 days His Grace Thomas Pelham - Holles 1st Duke of Newcastle KG PC FRS ( 1693 -- 1768 ) 16 March 1754 16 November 1756 1754 First Lord of the Treasury Leader of the House of Lords Whig Newcastle I 7002977000000000000 ♠ 2 years and 246 days His Grace William Cavendish 4th Duke of Devonshire KG PC ( 1720 -- 1764 ) 16 November 1756 25 June 1757 -- First Lord of the Treasury Leader of the House of Lords Whig Pitt -- Devonshire ​ 1757 Caretaker 7002222000000000000 ♠ 222 days His Grace Thomas Pelham - Holles 1st Duke of Newcastle KG PC FRS ( 1693 -- 1768 ) 2 July 1757 26 May 1762 1761 First Lord of the Treasury Leader of the House of Lords Whig Pitt -- Newcastle George III ( 1760 -- 1820 ) 7003179000000000000 ♠ 4 years and 329 days The Right Honourable John Stuart 3rd Earl of Bute KG PC ( 1713 -- 1792 ) 26 May 1762 8 April 1763 -- First Lord of the Treasury Leader of the House of Lords Tory Bute ( Tory -- Whig ) 7002318000000000000 ♠ 318 days The Right Honourable George Grenville MP for Buckingham ( 1712 -- 1770 ) 16 April 1763 13 July 1765 -- First Lord of the Treasury Chancellor of the Exchequer Leader of the House of Commons Whig ( Grenvillite ) Grenville ( Whig -- Tory ) 7002820000000000000 ♠ 2 years and 89 days The Most Honourable Charles Watson - Wentworth 2nd Marquess of Rockingham KG PC FRS ( 1730 -- 1782 ) 13 July 1765 30 July 1766 -- First Lord of the Treasury Leader of the House of Lords Whig ( Rockinghamite ) Rockingham I 7002383000000000000 ♠ 1 year and 18 days The Right Honourable William Pitt 1st Earl of Chatham PC FRS MP for Bath ( 1708 -- 1778 ) 30 July 1766 14 October 1768 1768 Lord Keeper of the Privy Seal Whig ( Chathamite ) Chatham ( Whig -- Tory ) 7002808000000000000 ♠ 2 years and 77 days His Grace Augustus FitzRoy 3rd Duke of Grafton KG PC ( 1735 -- 1811 ) 14 October 1768 28 January 1770 -- First Lord of the Treasury Leader of the House of Lords Whig ( Chathamite ) Grafton 7002472000000000000 ♠ 1 year and 107 days The Right Honourable Frederick North Lord North KG MP for Banbury ( 1732 -- 1792 ) 28 January 1770 22 March 1782 First Lord of the Treasury Chancellor of the Exchequer Leader of the House of Commons Tory ( Northite ) North 1774 1780 7003443700000000000 ♠ 12 years and 54 days The Most Honourable Charles Watson - Wentworth 2nd Marquess of Rockingham KG PC FRS ( 1730 -- 1782 ) 27 March 1782 1 July 1782 -- First Lord of the Treasury Leader of the House of Lords Whig ( Rockinghamite ) Rockingham II 7001970000000000000 ♠ 97 days The Right Honourable William Petty 2nd Earl of Shelburne KG PC ( 1737 -- 1805 ) 4 July 1782 2 April 1783 -- First Lord of the Treasury Leader of the House of Lords Whig ( Chathamite ) Shelburne ( Whig -- Tory ) 7002273000000000000 ♠ 273 days His Grace William Cavendish - Bentinck 3rd Duke of Portland PC FRS ( 1738 -- 1809 ) 2 April 1783 19 December 1783 -- First Lord of the Treasury Leader of the House of Lords Whig Fox -- North 7002262000000000000 ♠ 262 days The Right Honourable William Pitt the Younger MP for Appleby -- Cambridge University ( 1759 -- 1806 ) 19 December 1783 14 March 1801 First Lord of the Treasury Chancellor of the Exchequer Leader of the House of Commons Tory ( Pittite ) Pitt I 1784 1790 1796 7003629500000000000 ♠ 17 years and 86 days The Right Honourable Henry Addington MP for Devizes ( 1757 -- 1844 ) 17 March 1801 10 May 1804 1801 First Lord of the Treasury Chancellor of the Exchequer Leader of the House of Commons Tory ( Pittite ) Addington 1802 7003115100000000000 ♠ 3 years and 55 days The Right Honourable William Pitt the Younger MP for Cambridge University ( 1759 -- 1806 ) 10 May 1804 23 January 1806 -- First Lord of the Treasury Chancellor of the Exchequer Leader of the House of Commons Tory ( Pittite ) Pitt II 7002624000000000000 ♠ 1 year and 259 days The Right Honourable William Grenville 1st Baron Grenville PC PC ( Ire ) ( 1759 -- 1834 ) 11 February 1806 31 March 1807 First Lord of the Treasury Leader of the House of Lords Whig All the Talents ( Whig -- Tory ) 1806 7002414000000000000 ♠ 1 year and 49 days His Grace William Cavendish - Bentinck 3rd Duke of Portland KG PC FRS ( 1738 -- 1809 ) 31 March 1807 4 October 1809 First Lord of the Treasury Tory ( Pittite ) Portland II 1807 7002919000000000000 ♠ 2 years and 188 days The Right Honourable Spencer Perceval KC MP for Northampton ( 1762 -- 1812 ) 4 October 1809 11 May 1812 -- First Lord of the Treasury Chancellor of the Exchequer Chancellor of the Duchy of Lancaster Leader of the House of Commons Tory ( Pittite ) Perceval 7002951000000000000 ♠ 2 years and 221 days The Right Honourable Robert Jenkinson 2nd Earl of Liverpool KG PC FRS ( 1770 -- 1828 ) 8 June 1812 9 April 1827 1812 First Lord of the Treasury Leader of the House of Lords Tory ( Pittite ) Liverpool 1818 1820 George IV ( 1820 -- 1830 ) 1826 7003541900000000000 ♠ 14 years and 306 days The Right Honourable George Canning FRS MP for Seaford ( 1770 -- 1827 ) 10 April 1827 8 August 1827 -- First Lord of the Treasury Chancellor of the Exchequer Leader of the House of Commons Tory ( Canningite ) Canning ( Canningite -- Whig ) 7002121000000000000 ♠ 121 days The Right Honourable Frederick John Robinson 1st Viscount Goderich PC ( 1782 -- 1859 ) 31 August 1827 21 January 1828 -- First Lord of the Treasury Leader of the House of Lords Tory ( Canningite ) Goderich 7002144000000000000 ♠ 144 days Field Marshal His Grace Arthur Wellesley 1st Duke of Wellington KG GCB GCH PC ( 1769 -- 1852 ) 22 January 1828 16 November 1830 First Lord of the Treasury Leader of the House of Lords Tory Wellington -- Peel 1830 William IV ( 1830 -- 1837 ) 7003103000000000000 ♠ 2 years and 299 days The Right Honourable Charles Grey 2nd Earl Grey KG PC ( 1764 -- 1845 ) 22 November 1830 9 July 1834 First Lord of the Treasury Leader of the House of Lords Whig Grey 1831 1832 7003132600000000000 ♠ 3 years and 230 days The Right Honourable William Lamb 2nd Viscount Melbourne PC ( 1779 -- 1848 ) 16 July 1834 14 November 1834 -- First Lord of the Treasury Leader of the House of Lords Whig Melbourne I 7002122000000000000 ♠ 122 days Field Marshal His Grace Arthur Wellesley 1st Duke of Wellington KG GCB GCH PC ( 1769 -- 1852 ) 14 November 1834 10 December 1834 -- First Lord of the Treasury Sec. of State for the Home Department Sec. of State for Foreign Affairs Sec. of State for War and the Colonies Leader of the House of Lords Tory Wellington Caretaker 7001270000000000000 ♠ 27 days The Right Honourable Sir Robert Peel Bt FRS MP for Tamworth ( 1788 -- 1850 ) 10 December 1834 8 April 1835 -- First Lord of the Treasury Chancellor of the Exchequer Leader of the House of Commons Conservative Peel I 7002120000000000000 ♠ 120 days The Right Honourable William Lamb 2nd Viscount Melbourne PC FRS ( 1779 -- 1848 ) 18 April 1835 30 August 1841 1835 First Lord of the Treasury Leader of the House of Lords Whig Melbourne II 1837 Victoria ( 1837 -- 1901 ) 7003232700000000000 ♠ 6 years and 135 days The Right Honourable Sir Robert Peel Bt FRS MP for Tamworth ( 1788 -- 1850 ) 30 August 1841 29 June 1846 1841 First Lord of the Treasury Leader of the House of Commons Conservative Peel II 7003176500000000000 ♠ 4 years and 304 days The Right Honourable Lord John Russell FRS MP for the City of London ( 1792 -- 1878 ) 30 June 1846 21 February 1852 1847 First Lord of the Treasury Leader of the House of Commons Whig Russell I 7003206300000000000 ♠ 5 years and 237 days The Right Honourable Edward Smith - Stanley 14th Earl of Derby PC ( 1799 -- 1869 ) 23 February 1852 17 December 1852 First Lord of the Treasury Leader of the House of Lords Conservative Who ? Who ? 1852 7002299000000000000 ♠ 299 days The Right Honourable George Hamilton - Gordon 4th Earl of Aberdeen KT FRSE FRS PC FSA Scot ( 1784 -- 1860 ) 19 December 1852 30 January 1855 -- First Lord of the Treasury Leader of the House of Lords Peelite Aberdeen ( Peelite -- Whig ) 7002773000000000000 ♠ 2 years and 43 days The Right Honourable Henry John Temple 3rd Viscount Palmerston KG GCB PC FRS MP for Tiverton ( 1784 -- 1865 ) 6 February 1855 19 February 1858 First Lord of the Treasury Leader of the House of Commons Whig Palmerston I 1857 7003111000000000000 ♠ 3 years and 14 days The Right Honourable Edward Smith - Stanley 14th Earl of Derby PC ( 1799 -- 1869 ) 20 February 1858 11 June 1859 -- First Lord of the Treasury Leader of the House of Lords Conservative Derby -- Disraeli II 7002477000000000000 ♠ 1 year and 112 days The Right Honourable Henry John Temple 3rd Viscount Palmerston KG GCB PC FRS MP for Tiverton ( 1784 -- 1865 ) 12 June 1859 18 October 1865 1859 First Lord of the Treasury Leader of the House of Commons Liberal Palmerston II 1865 7003232100000000000 ♠ 6 years and 129 days The Right Honourable John Russell 1st Earl Russell KG PC FRS ( 1792 -- 1878 ) 29 October 1865 26 June 1866 -- First Lord of the Treasury Leader of the House of Lords Liberal Russell II 7002241000000000000 ♠ 241 days The Right Honourable Edward Smith - Stanley 14th Earl of Derby KG PC ( 1799 -- 1869 ) 28 June 1866 25 February 1868 -- First Lord of the Treasury Leader of the House of Lords Conservative Derby -- Disraeli III 7002608000000000000 ♠ 1 year and 243 days The Right Honourable Benjamin Disraeli MP for Buckinghamshire ( 1804 -- 1881 ) See also § Premierships : 1 First Lord of the Treasury Leader of the House of Commons Conservative 27 February 1868 1 December 1868 -- 7002279000000000000 ♠ 279 days The Right Honourable William Ewart Gladstone FSS MP for Greenwich ( 1809 -- 1898 ) See also § Premierships : 2 First Lord of the Treasury Leader of the House of Commons Chancellor of the Exchequer Liberal Gladstone I 3 December 1868 17 February 1874 1868 7003190300000000000 ♠ 5 years and 77 days The Right Honourable Benjamin Disraeli 1st Earl of Beaconsfield KG PC FRS MP for Buckinghamshire ( 1804 -- 1881 ) See also § Premierships : 1 First Lord of the Treasury Leader of the House of Commons Lord Keeper of the Privy Seal Leader of the House of Lords Conservative Disraeli II 20 February 1874 21 April 1880 1874 7003225300000000000 ♠ 6 years and 62 days The Right Honourable William Ewart Gladstone FRS FSS MP for Midlothian ( 1809 -- 1898 ) See also § Premierships : 2 First Lord of the Treasury Leader of the House of Commons Chancellor of the Exchequer Liberal Gladstone II 23 April 1880 9 June 1885 1880 7003187400000000000 ♠ 5 years and 48 days The Most Honourable Robert Gascoyne - Cecil 3rd Marquess of Salisbury KG PC FRS DL ( 1830 -- 1903 ) 23 June 1885 28 January 1886 -- Sec. of State for Foreign Affairs Leader of the House of Lords Conservative Salisbury I 7002220000000000000 ♠ 220 days The Right Honourable William Ewart Gladstone FRS FSS MP for Midlothian ( 1809 -- 1898 ) See also § Premierships : 2 First Lord of the Treasury Lord Keeper of the Privy Seal Leader of the House of Commons Liberal Gladstone III 1 February 1886 20 July 1886 1885 7002170000000000000 ♠ 170 days The Most Honourable Robert Gascoyne - Cecil 3rd Marquess of Salisbury KG PC FRS DL ( 1830 -- 1903 ) 25 July 1886 11 August 1892 1886 Leader of the House of Lords First Lord of the Treasury Sec. of State for Foreign Affairs Conservative Salisbury II 7003221000000000000 ♠ 6 years and 18 days The Right Honourable William Ewart Gladstone FRS FSS MP for Midlothian ( 1809 -- 1898 ) See also § Premierships : 2 First Lord of the Treasury Lord Keeper of the Privy Seal Leader of the House of Commons Liberal Gladstone IV 15 August 1892 2 March 1894 1892 7002565000000000000 ♠ 1 year and 200 days The Right Honourable Archibald Primrose 5th Earl of Rosebery KG PC FRS ( 1847 -- 1929 ) 5 March 1894 22 June 1895 -- First Lord of the Treasury Lord President of the Council Leader of the House of Lords Liberal Rosebery 7002475000000000000 ♠ 1 year and 110 days The Most Honourable Robert Gascoyne - Cecil 3rd Marquess of Salisbury KG PC FRS DL ( 1830 -- 1903 ) 25 June 1895 11 July 1902 Leader of the House of Lords Sec. of State for Foreign Affairs Lord Keeper of the Privy Seal Conservative Salisbury III ( Con. -- Lib. U . ) 1895 1900 7003257300000000000 ♠ 7 years and 17 days Edward VII ( 1901 -- 1910 ) The Right Honourable Arthur Balfour FRS DL MP for Manchester East ( 1848 -- 1930 ) 11 July 1902 5 December 1905 -- First Lord of the Treasury Leader of the House of Commons Conservative Balfour 7003124400000000000 ♠ 3 years and 148 days The Right Honourable Sir Henry Campbell - Bannerman GCB MP for Stirling Burghs ( 1836 -- 1908 ) 5 December 1905 7 April 1908 First Lord of the Treasury Leader of the House of Commons Liberal Campbell - Bannerman 1906 7002855000000000000 ♠ 2 years and 125 days The Right Honourable Herbert Henry Asquith KC FRS MP for East Fife ( 1852 -- 1928 ) 7 April 1908 5 December 1916 First Lord of the Treasury Leader of the House of Commons Sec. of State for War Liberal Asquith I 1910 ​ 1910 George V ( 1910 -- 1936 ) ​ Asquith II ( Lib. -- Con. -- et al . ) 7003316500000000000 ♠ 8 years and 243 days The Right Honourable David Lloyd George OM MP for Caernarvon Boroughs ( 1863 -- 1945 ) 6 December 1916 19 October 1922 First Lord of the Treasury Liberal Lloyd George War 1918 Lloyd George II ( Lib. -- Con . ) 7003214400000000000 ♠ 5 years and 318 days The Right Honourable Bonar Law MP for Glasgow Central ( 1858 -- 1923 ) 23 October 1922 20 May 1923 1922 First Lord of the Treasury Leader of the House of Commons Conservative ( Scot. U . ) Law 7002210000000000000 ♠ 210 days The Right Honourable Stanley Baldwin JP MP for Bewdley ( 1867 -- 1947 ) 23 May 1923 16 January 1924 -- First Lord of the Treasury Leader of the House of Commons Chancellor of the Exchequer Conservative Baldwin I 7002239000000000000 ♠ 239 days The Right Honourable Ramsay MacDonald MP for Aberavon ( 1866 -- 1937 ) 22 January 1924 4 November 1924 1923 First Lord of the Treasury Leader of the House of Commons Sec. of State for Foreign Affairs Labour MacDonald I 7002288000000000000 ♠ 288 days The Right Honourable Stanley Baldwin JP FRS MP for Bewdley ( 1867 -- 1947 ) 4 November 1924 5 June 1929 1924 First Lord of the Treasury Leader of the House of Commons Conservative Baldwin II 7003167500000000000 ♠ 4 years and 214 days The Right Honourable Ramsay MacDonald FRS MP for Seaham ( 1866 -- 1937 ) 5 June 1929 7 June 1935 1929 First Lord of the Treasury Leader of the House of Commons Labour MacDonald II National Labour National I ( N. Lab. -- Con. -- et al . ) 1931 National II 7003219400000000000 ♠ 6 years and 3 days ​ The Right Honourable Stanley Baldwin JP FRS MP for Bewdley ( 1867 -- 1947 ) 7 June 1935 28 May 1937 1935 First Lord of the Treasury Leader of the House of Commons Conservative National III ( Con. -- N. Lab. -- et al . ) Edward VIII ( 1936 ) George VI ( 1936 -- 1952 ) 7002722000000000000 ♠ 1 year and 356 days The Right Honourable Neville Chamberlain FRS MP for Birmingham Edgbaston ( 1869 -- 1940 ) 28 May 1937 10 May 1940 -- First Lord of the Treasury Leader of the House of Commons Conservative National IV ​ Chamberlain War 7003107900000000000 ♠ 2 years and 349 days The Right Honourable Winston Churchill CH TD PCc FRS MP for Epping ( 1874 -- 1965 ) 10 May 1940 26 July 1945 -- First Lord of the Treasury Minister of Defence Leader of the House of Commons Conservative Churchill War ( All parties ) ​ Churchill Caretaker ( Con. -- Lib. N . ) 7003190400000000000 ♠ 5 years and 78 days The Right Honourable Clement Attlee CH FRS MP for Limehouse -- Walthamstow West ( 1883 -- 1967 ) 26 July 1945 26 October 1951 1945 First Lord of the Treasury Minister of Defence Labour Attlee I 1950 Attlee II 7003228400000000000 ♠ 6 years and 93 days The Right Honourable Sir Winston Churchill KG OM CH TD PCc DL FRS RA MP for Woodford ( 1874 -- 1965 ) 26 October 1951 6 April 1955 1951 First Lord of the Treasury Minister of Defence Conservative Churchill III 7003125900000000000 ♠ 3 years and 163 days Elizabeth II ( 1952 -- present ) The Right Honourable Sir Anthony Eden KG MC MP for Warwick & Leamington ( 1897 -- 1977 ) 6 April 1955 10 January 1957 1955 First Lord of the Treasury Conservative Eden 7002646000000000000 ♠ 1 year and 280 days The Right Honourable Harold Macmillan FRS MP for Bromley ( 1894 -- 1986 ) 10 January 1957 19 October 1963 1959 First Lord of the Treasury Conservative Macmillan I ​ Macmillan II 7003247400000000000 ♠ 6 years and 283 days The Right Honourable Sir Alec Douglas - Home KT MP for Kinross & Western Perthshire ( 1903 -- 1995 ) 19 October 1963 16 October 1964 -- First Lord of the Treasury Conservative ( Scot. U . ) Douglas - Home 7002364000000000000 ♠ 364 days The Right Honourable Harold Wilson OBE FRS MP for Huyton ( 1916 -- 1995 ) 16 October 1964 19 June 1970 1964 First Lord of the Treasury Minister for the Civil Service Labour Wilson I 1966 Wilson II 7003207300000000000 ♠ 5 years and 247 days The Right Honourable Edward Heath MBE MP for Bexley -- Sidcup ( 1916 -- 2005 ) 19 June 1970 4 March 1970 First Lord of the Treasury Minister for the Civil Service Conservative Heath 7003135500000000000 ♠ 3 years and 259 days The Right Honourable Harold Wilson OBE FRS MP for Huyton ( 1916 -- 1995 ) 4 March 5 April 1976 First Lord of the Treasury Minister for the Civil Service Labour Wilson III Wilson IV 7002764000000000000 ♠ 2 years and 33 days The Right Honourable James Callaghan MP for Cardiff South East ( 1912 -- 2005 ) 5 April 1976 4 May 1979 -- First Lord of the Treasury Minister for the Civil Service Labour Callaghan 7003112500000000000 ♠ 3 years and 30 days The Right Honourable Margaret Thatcher FRS FRIC MP for Finchley ( 1925 -- 2013 ) See also § Premierships :3 First Lord of the Treasury Minister for the Civil Service Conservative Thatcher I 4 May 1979 28 November 1990 1979 Thatcher II Thatcher III 7003422700000000000 ♠ 11 years and 209 days The Right Honourable John Major MP for Huntingdon ( born 1943 ) 28 November 1990 2 May 1997 First Lord of the Treasury Minister for the Civil Service Conservative Major I ​ Major II 7003234800000000000 ♠ 6 years and 156 days The Right Honourable Tony Blair MP for Sedgefield ( born 1953 ) See also § Premierships : 4 First Lord of the Treasury Minister for the Civil Service Labour Blair I 2 May 1997 27 June 2007 1997 Blair II 2005 Blair III 7003370900000000000 ♠ 10 years and 57 days The Right Honourable Gordon Brown MP for Kirkcaldy & Cowdenbeath ( born 1951 ) See also § Premierships : 5 First Lord of the Treasury Minister for the Civil Service Labour Brown 27 June 2007 11 May -- 7003105000000000000 ♠ 2 years and 319 days The Right Honourable David Cameron MP for Witney ( born 1966 ) See also § Premierships : 6 First Lord of the Treasury Minister for the Civil Service Conservative Cameron -- Clegg ( Con. -- L.D. ) 11 May 13 July 2016 2015 Cameron II 7003225600000000000 ♠ 6 years and 64 days The Right Honourable Theresa May MP for Maidenhead ( born 1956 ) See also § Premierships : 7 First Lord of the Treasury Minister for the Civil Service Conservative May I 13 July 2016 Incumbent 2017 May II 7002558000000000000 ♠ 1 year and 193 days
  Fact 0.0: Robert Walpole, 1st Earl of Orford, was MP for King's Lynn and lived from 1676 to 1745.
  Fact 0.1: Robert Walpole was Prime Minister from 4 April 1721 to 11 February 1742, for 20 years and 314 days.
  Fact 0.2: Robert Walpole was a Whig.
  Fact 0.3: Spencer Compton, 1st Earl of Wilmington, lived from 1673 to 1743.
  Fact 0.4: Spencer Compton was Prime Minister from 16 February 1742 to 2 July 1743, for 1 year and 137 days.
  Fact 0.5: Spencer Compton was a Whig.
  Fact 0.6: Henry Pelham was MP for Sussex and lived from 1694 to 1754.
  Fact 0.7: Henry Pelham was Prime Minister from 27 August 1743 to 6 March 1754, for 10 years and 192 days.
  Fact 0.8: Henry Pelham was a Whig.
  Fact 0.9: Thomas Pelham-Holles, 1st Duke of Newcastle, lived from 1693 to 1768.
  Fact 0.10: Thomas Pelham-Holles was Prime Minister from 16 March 1754 to 16 November 1756, for 2 years and 246 days.
  Fact 0.11: Thomas Pelham-Holles was a Whig.
  Fact 0.12: William Cavendish, 4th Duke of Devonshire, lived from 1720 to 1764.
  Fact 0.13: William Cavendish was Prime Minister from 16 November 1756 to 25 June 1757, for 222 days.
  Fact 0.14: William Cavendish's ministry was a caretaker ministry in 1757.
  Fact 0.15: William Cavendish was a Whig.
  Fact 0.16: Thomas Pelham-Holles was Prime Minister again from 2 July 1757 to 26 May 1762, for 4 years and 329 days.
  Fact 0.17: Thomas Pelham-Holles was a Whig in his second term.
  Fact 0.18: John Stuart, 3rd Earl of Bute, lived from 1713 to 1792.
  Fact 0.19: John Stuart was Prime Minister from 26 May 1762 to 8 April 1763, for 318 days.
  Fact 0.20: John Stuart was a Tory.
  Fact 0.21: George Grenville was MP for Buckingham and lived from 1712 to 1770.
  Fact 0.22: George Grenville was Prime Minister from 16 April 1763 to 13 July 1765, for 2 years and 89 days.
  Fact 0.23: George Grenville was a Whig, in the Grenvillite faction.
  Fact 0.24: Charles Watson-Wentworth, 2nd Marquess of Rockingham, lived from 1730 to 1782.
  Fact 0.25: Charles Watson-Wentworth was Prime Minister from 13 July 1765 to 30 July 1766, for 1 year and 18 days.
  Fact 0.26: Charles Watson-Wentworth was a Whig, in the Rockinghamite faction.
  Fact 0.27: William Pitt, 1st Earl of Chatham, was MP for Bath and lived from 1708 to 1778.
  Fact 0.28: William Pitt, 1st Earl of Chatham, was Prime Minister from 30 July 1766 to 14 October 1768, for 2 years and 77 days.
  Fact 0.29: William Pitt, 1st Earl of Chatham, was a Whig, in the Chathamite faction.
  Fact 0.30: Augustus FitzRoy, 3rd Duke of Grafton, lived from 1735 to 1811.
  Fact 0.31: Augustus FitzRoy was Prime Minister from 14 October 1768 to 28 January 1770, for 1 year and 107 days.
  Fact 0.32: Augustus FitzRoy was a Whig, in the Chathamite faction.
  Fact 0.33: Frederick North, Lord North, was MP for Banbury and lived from 1732 to 1792.
  Fact 0.34: Frederick North was Prime Minister from 28 January 1770 to 22 March 1782, for 12 years and 54 days.
  Fact 0.35: Frederick North was a Tory, in the Northite faction.
  Fact 0.36: Charles Watson-Wentworth was Prime Minister again from 27 March 1782 to 1 July 1782, for 97 days.
  Fact 0.37: Charles Watson-Wentworth was a Whig in his second term, in the Rockinghamite faction.
  Fact 0.38: William Petty, 2nd Earl of Shelburne, lived from 1737 to 1805.
  Fact 0.39: William Petty was Prime Minister from 4 July 1782 to 2 April 1783, for 273 days.
  Fact 0.40: William Petty was a Whig, in the Chathamite faction.
  Fact 0.41: William Cavendish-Bentinck, 3rd Duke of Portland, lived from 1738 to 1809.
  Fact 0.42: William Cavendish-Bentinck was Prime Minister from 2 April 1783 to 19 December 1783, for 262 days.
  Fact 0.43: William Cavendish-Bentinck was a Whig.
  Fact 0.44: William Pitt the Younger was MP for Appleby and Cambridge University, and lived from 1759 to 1806.
  Fact 0.45: William Pitt the Younger was Prime Minister from 19 December 1783 to 14 March 1801, for 17 years and 86 days.
  Fact 0.46: William Pitt the Younger was a Tory, in the Pittite faction.
  Fact 0.47: Henry Addington was MP for Devizes and lived from 1757 to 1844.
  Fact 0.48: Henry Addington was Prime Minister from 17 March 1801 to 10 May 1804, for 3 years and 55 days.
  Fact 0.49: Henry Addington was a Tory, in the Pittite faction.
  Fact 0.50: William Pitt the Younger was Prime Minister again from 10 May 1804 to 23 January 1806, for 1 year and 259 days.
  Fact 0.51: William Pitt the Younger was a Tory, in the Pittite faction, in his second term.
  Fact 0.52: William Grenville, 1st Baron Grenville, lived from 1759 to 1834.
  Fact 0.53: William Grenville was Prime Minister from 11 February 1806 to 31 March 1807, for 1 year and 49 days.
  Fact 0.54: William Grenville was a Whig, in the All the Talents ministry.
  Fact 0.55: William Cavendish-Bentinck was Prime Minister again from 31 March 1807 to 4 October 1809, for 2 years and 188 days.
  Fact 0.56: William Cavendish-Bentinck was a Tory, in the Pittite faction, in his second term.
  Fact 0.57: Spencer Perceval was MP for Northampton and lived from 1762 to 1812.
  Fact 0.58: Spencer Perceval was Prime Minister from 4 October 1809 to 11 May 1812, for 2 years and 221 days.
  Fact 0.59: Spencer Perceval was a Tory, in the Pittite faction.
  Fact 0.60: Robert Jenkinson, 2nd Earl of Liverpool, lived from 1770 to 1828.
  Fact 0.61: Robert Jenkinson was Prime Minister from 8 June 1812 to 9 April 1827, for 14 years and 306 days.
  Fact 0.62: Robert Jenkinson was a Tory, in the Pittite faction.
  Fact 0.63: George Canning was MP for Seaford and lived from 1770 to 1827.
  Fact 0.64: George Canning was Prime Minister from 10 April 1827 to 8 August 1827, for 121 days.
  Fact 0.65: George Canning was a Tory, in the Canningite faction.
  Fact 0.66: Frederick John Robinson, 1st Viscount Goderich, lived from 1782 to 1859.
  Fact 0.67: Frederick John Robinson was Prime Minister from 31 August 1827 to 21 January 1828, for 144 days.
  Fact 0.68: Frederick John Robinson was a Tory, in the Canningite faction.
  Fact 0.69: Arthur Wellesley, 1st Duke of Wellington, lived from 1769 to 1852.
  Fact 0.70: Arthur Wellesley was Prime Minister from 22 January 1828 to 16 November 1830, for 2 years and 299 days.
  Fact 0.71: Arthur Wellesley was a Tory.
  Fact 0.72: Charles Grey, 2nd Earl Grey, lived from 1764 to 1845.
  Fact 0.73: Charles Grey was Prime Minister from 22 November 1830 to 9 July 1834, for 3 years and 230 days.
  Fact 0.74: Charles Grey was a Whig.
  Fact 0.75: William Lamb, 2nd Viscount Melbourne, lived from 1779 to 1848.
  Fact 0.76: William Lamb was Prime Minister from 16 July 1834 to 14 November 1834, for 122 days.
  Fact 0.77: William Lamb was a Whig.
  Fact 0.78: Arthur Wellesley was Prime Minister again from 14 November 1834 to 10 December 1834, for 27 days.
  Fact 0.79: Arthur Wellesley's second term was a caretaker ministry, and he was a Tory.
  Fact 0.80: Robert Peel was MP for Tamworth and lived from 1788 to 1850.
  Fact 0.81: Robert Peel was Prime Minister from 10 December 1834 to 8 April 1835, for 120 days.
  Fact 0.82: Robert Peel was a Conservative.
  Fact 0.83: William Lamb was Prime Minister again from 18 April 1835 to 30 August 1841, for 6 years and 135 days.
  Fact 0.84: William Lamb was a Whig in his second term.
  Fact 0.85: Robert Peel was Prime Minister again from 30 August 1841 to 29 June 1846, for 4 years and 304 days.
  Fact 0.86: Robert Peel was a Conservative in his second term.
  Fact 0.87: Lord John Russell was MP for the City of London and lived from 1792 to 1878.
  Fact 0.88: Lord John Russell was Prime Minister from 30 June 1846 to 21 February 1852, for 5 years and 237 days.
  Fact 0.89: Lord John Russell was a Whig.
  Fact 0.90: Edward Smith-Stanley, 14th Earl of Derby, lived from 1799 to 1869.
  Fact 0.91: Edward Smith-Stanley was Prime Minister from 23 February 1852 to 17 December 1852, for 299 days.
  Fact 0.92: Edward Smith-Stanley was a Conservative.
  Fact 0.93: George Hamilton-Gordon, 4th Earl of Aberdeen, lived from 1784 to 1860.
  Fact 0.94: George Hamilton-Gordon was Prime Minister from 19 December 1852 to 30 January 1855, for 2 years and 43 days.
  Fact 0.95: George Hamilton-Gordon was a Peelite.
  Fact 0.96: Henry John Temple, 3rd Viscount Palmerston, was MP for Tiverton and lived from 1784 to 1865.
  Fact 0.97: Henry John Temple was Prime Minister from 6 February 1855 to 19 February 1858, for 3 years and 14 days.
  Fact 0.98: Henry John Temple was a Whig.
  Fact 0.99: Edward Smith-Stanley was Prime Minister again from 20 February 1858 to 11 June 1859, for 1 year and 112 days.
  Fact 0.100: Edward Smith-Stanley was a Conservative in his second term.
  Fact 0.101: Henry John Temple was Prime Minister again from 12 June 1859 to 18 October 1865, for 6 years and 129 days.
  Fact 0.102: Henry John Temple was a Liberal in his second term.
  Fact 0.103: John Russell, 1st Earl Russell, lived from 1792 to 1878.
  Fact 0.104: John Russell, 1st Earl Russell, was Prime Minister from 29 October 1865 to 26 June 1866, for 241 days.
  Fact 0.105: John Russell, 1st Earl Russell, was a Liberal.
  Fact 0.106: Edward Smith-Stanley was Prime Minister again from 28 June 1866 to 25 February 1868, for 1 year and 243 days.
  Fact 0.107: Edward Smith-Stanley was a Conservative in his third term.
  Fact 0.108: Benjamin Disraeli was MP for Buckinghamshire and lived from 1804 to 1881.
  Fact 0.109: Benjamin Disraeli was Prime Minister from 27 February 1868 to 1 December 1868, for 279 days.
  Fact 0.110: Benjamin Disraeli was a Conservative.
  Fact 0.111: William Ewart Gladstone was MP for Greenwich and lived from 1809 to 1898.
  Fact 0.112: William Ewart Gladstone was Prime Minister from 3 December 1868 to 17 February 1874, for 5 years and 77 days.
  Fact 0.113: William Ewart Gladstone was a Liberal.
  Fact 0.114: Benjamin Disraeli, 1st Earl of Beaconsfield, was Prime Minister again from 20 February 1874 to 21 April 1880, for 6 years and 62 days.
  Fact 0.115: Benjamin Disraeli was a Conservative in his second term.
  Fact 0.116: William Ewart Gladstone was MP for Midlothian and was Prime Minister again from 23 April 1880 to 9 June 1885, for 5 years and 48 days.
  Fact 0.117: William Ewart Gladstone was a Liberal in his second term.
  Fact 0.118: Robert Gascoyne-Cecil, 3rd Marquess of Salisbury, lived from 1830 to 1903.
  Fact 0.119: Robert Gascoyne-Cecil was Prime Minister from 23 June 1885 to 28 January 1886, for 220 days.
  Fact 0.120: Robert Gascoyne-Cecil was a Conservative.
  Fact 0.121: William Ewart Gladstone was Prime Minister again from 1 February 1886 to 20 July 1886, for 170 days.
  Fact 0.122: William Ewart Gladstone was a Liberal in his third term.
  Fact 0.123: Robert Gascoyne-Cecil was Prime Minister again from 25 July 1886 to 11 August 1892, for 6 years and 18 days.
  Fact 0.124: Robert Gascoyne-Cecil was a Conservative in his second term.
  Fact 0.125: William Ewart Gladstone was Prime Minister again from 15 August 1892 to 2 March 1894, for 1 year and 200 days.
  Fact 0.126: William Ewart Gladstone was a Liberal in his fourth term.
  Fact 0.127: Archibald Primrose, 5th Earl of Rosebery, lived from 1847 to 1929.
  Fact 0.128: Archibald Primrose was Prime Minister from 5 March 1894 to 22 June 1895, for 1 year and 110 days.
  Fact 0.129: Archibald Primrose was a Liberal.
  Fact 0.130: Robert Gascoyne-Cecil was Prime Minister again from 25 June 1895 to 11 July 1902, for 7 years and 17 days.
  Fact 0.131: Robert Gascoyne-Cecil's third term was labelled Con. and Lib. U. in the table.
  Fact 0.132: Arthur Balfour was MP for Manchester East and lived from 1848 to 1930.
  Fact 0.133: Arthur Balfour was Prime Minister from 11 July 1902 to 5 December 1905, for 3 years and 148 days.
  Fact 0.134: Arthur Balfour was a Conservative.
  Fact 0.135: Henry Campbell-Bannerman was MP for Stirling Burghs and lived from 1836 to 1908.
  Fact 0.136: Henry Campbell-Bannerman was Prime Minister from 5 December 1905 to 7 April 1908, for 2 years and 125 days.
  Fact 0.137: Henry Campbell-Bannerman was a Liberal.
  Fact 0.138: Herbert Henry Asquith was MP for East Fife and lived from 1852 to 1928.
  Fact 0.139: Herbert Henry Asquith was Prime Minister from 7 April 1908 to 5 December 1916, for 8 years and 243 days.
  Fact 0.140: Herbert Henry Asquith was a Liberal.
  Fact 0.141: David Lloyd George was MP for Caernarvon Boroughs and lived from 1863 to 1945.
  Fact 0.142: David Lloyd George was Prime Minister from 6 December 1916 to 19 October 1922, for 5 years and 318 days.
  Fact 0.143: David Lloyd George was a Liberal.
  Fact 0.144: Bonar Law was MP for Glasgow Central and lived from 1858 to 1923.
  Fact 0.145: Bonar Law was Prime Minister from 23 October 1922 to 20 May 1923, for 210 days.
  Fact 0.146: Bonar Law was a Conservative.
  Fact 0.147: Stanley Baldwin was MP for Bewdley and lived from 1867 to 1947.
  Fact 0.148: Stanley Baldwin was Prime Minister from 23 May 1923 to 16 January 1924, for 239 days.
  Fact 0.149: Stanley Baldwin was a Conservative.
  Fact 0.150: Ramsay MacDonald was MP for Aberavon and lived from 1866 to 1937.
  Fact 0.151: Ramsay MacDonald was Prime Minister from 22 January 1924 to 4 November 1924, for 288 days.
  Fact 0.152: Ramsay MacDonald was a Labour politician.
  Fact 0.153: Stanley Baldwin was Prime Minister again from 4 November 1924 to 5 June 1929, for 4 years and 214 days.
  Fact 0.154: Stanley Baldwin was a Conservative in his second term.
  Fact 0.155: Ramsay MacDonald was MP for Seaham and was Prime Minister again from 5 June 1929 to 7 June 1935, for 6 years and 3 days.
  Fact 0.156: Ramsay MacDonald was a Labour politician in his second term.
  Fact 0.157: Stanley Baldwin was Prime Minister again from 7 June 1935 to 28 May 1937, for 1 year and 356 days.
  Fact 0.158: Stanley Baldwin was a Conservative in his third term.
  Fact 0.159: Neville Chamberlain was MP for Birmingham Edgbaston and lived from 1869 to 1940.
  Fact 0.160: Neville Chamberlain was Prime Minister from 28 May 1937 to 10 May 1940, for 2 years and 349 days.
  Fact 0.161: Neville Chamberlain was a Conservative.
  Fact 0.162: Winston Churchill was MP for Epping and lived from 1874 to 1965.
  Fact 0.163: Winston Churchill was Prime Minister from 10 May 1940 to 26 July 1945, for 5 years and 78 days.
  Fact 0.164: Winston Churchill was a Conservative.
  Fact 0.165: Clement Attlee was MP for Limehouse and Walthamstow West and lived from 1883 to 1967.
  Fact 0.166: Clement Attlee was Prime Minister from 26 July 1945 to 26 October 1951, for 6 years and 93 days.
  Fact 0.167: Clement Attlee was a Labour politician.
  Fact 0.168: Winston Churchill was MP for Woodford and was Prime Minister again from 26 October 1951 to 6 April 1955, for 3 years and 163 days.
  Fact 0.169: Winston Churchill was a Conservative in his second term.
  Fact 0.170: Anthony Eden was MP for Warwick and Leamington and lived from 1897 to 1977.
  Fact 0.171: Anthony Eden was Prime Minister from 6 April 1955 to 10 January 1957, for 1 year and 280 days.
  Fact 0.172: Anthony Eden was a Conservative.
  Fact 0.173: Harold Macmillan was MP for Bromley and lived from 1894 to 1986.
  Fact 0.174: Harold Macmillan was Prime Minister from 10 January 1957 to 19 October 1963, for 6 years and 283 days.
  Fact 0.175: Harold Macmillan was a Conservative.
  Fact 0.176: Alec Douglas-Home was MP for Kinross and Western Perthshire and lived from 1903 to 1995.
  Fact 0.177: Alec Douglas-Home was Prime Minister from 19 October 1963 to 16 October 1964, for 364 days.
  Fact 0.178: Alec Douglas-Home was a Conservative, in the Scottish Unionist faction.
  Fact 0.179: Harold Wilson was MP for Huyton and lived from 1916 to 1995.
  Fact 0.180: Harold Wilson was Prime Minister from 16 October 1964 to 19 June 1970, for 5 years and 247 days.
  Fact 0.181: Harold Wilson was a Labour politician.
  Fact 0.182: Edward Heath was MP for Bexley and Sidcup and lived from 1916 to 2005.
  Fact 0.183: Edward Heath was Prime Minister from 19 June 1970 to 4 March 1970, as the table lists it, for 3 years and 259 days.
  Fact 0.184: Edward Heath was a Conservative.
  Fact 0.185: Harold Wilson was Prime Minister again from 4 March to 5 April 1976, for 2 years and 33 days.
  Fact 0.186: Harold Wilson was a Labour politician in his second term.
  Fact 0.187: James Callaghan was MP for Cardiff South East and lived from 1912 to 2005.
  Fact 0.188: James Callaghan was Prime Minister from 5 April 1976 to 4 May 1979, for 3 years and 30 days.
  Fact 0.189: James Callaghan was a Labour politician.
  Fact 0.190: Margaret Thatcher was MP for Finchley and lived from 1925 to 2013.
  Fact 0.191: Margaret Thatcher was Prime Minister from 4 May 1979 to 28 November 1990, for 11 years and 209 days.
  Fact 0.192: Margaret Thatcher was a Conservative.
  Fact 0.193: John Major was MP for Huntingdon and was born in 1943.
  Fact 0.194: John Major was Prime Minister from 28 November 1990 to 2 May 1997, for 6 years and 156 days.
  Fact 0.195: John Major was a Conservative.
  Fact 0.196: Tony Blair was MP for Sedgefield and was born in 1953.
  Fact 0.197: Tony Blair was Prime Minister from 2 May 1997 to 27 June 2007, for 10 years and 57 days.
  Fact 0.198: Tony Blair was a Labour politician.
  Fact 0.199: Gordon Brown was MP for Kirkcaldy and Cowdenbeath and was born in 1951.
  Fact 0.200: Gordon Brown was Prime Minister from 27 June 2007 to 11 May (the year is not shown in the table), for 2 years and 319 days.
  Fact 0.201: Gordon Brown was a Labour politician.
  Fact 0.202: David Cameron was MP for Witney and was born in 1966.
  Fact 0.203: David Cameron was Prime Minister from 11 May (the year is not shown in the table) to 13 July 2016, for 6 years and 64 days.
  Fact 0.204: David Cameron was a Conservative, in a coalition with the Liberal Democrats.
  Fact 0.205: Theresa May was MP for Maidenhead and was born in 1956.
  Fact 0.206: Theresa May has been Prime Minister since 13 July 2016.
  Fact 0.207: Theresa May's term was 1 year and 193 days as of the table.
  Fact 0.208: Theresa May was a Conservative.
  Fact 0.209: The monarch Elizabeth II reigned from 1952 to the present.
  Fact 0.210: The monarch George I reigned from 1714 to 1727.
  Fact 0.211: The monarch George II reigned from 1727 to 1760.
  Fact 0.212: The monarch George III reigned from 1760 to 1820.
  Fact 0.213: The monarch George IV reigned from 1820 to 1830.
  Fact 0.214: The monarch William IV reigned from 1830 to 1837.
  Fact 0.215: The monarch Victoria reigned from 1837 to 1901.
  Fact 0.216: The monarch Edward VII reigned from 1901 to 1910.
  Fact 0.217: The monarch George V reigned from 1910 to 1936.
  Fact 0.218: The monarch Edward VIII reigned in 1936.
  Fact 0.219: The monarch George VI reigned from 1936 to 1952.

Retrieved items:
<item 1>
List of Prime Ministers of the United Kingdom
2 years and 319 days The Right Honourable David Cameron MP for Witney ( born 1966 ) See also § Premierships : 6 First Lord of the Treasury Minister for the
Civil Service Conservative Cameron -- Clegg ( Con. -- L.D. ) 11 May 13 July 2016 2015 Cameron II 7003225600000000000 ♠ 6 years and 64 days The Right Honourable
Theresa May MP for Maidenhead ( born 1956 ) See also § Premierships : 7 First Lord of the Treasury Minister for the Civil Service Conservative May I 13 July
2016 Incumbent 2017 May II 7002558000000000000 ♠ 1 year and 193 days Addenda ( edit ) ^ 1 Whig majority in the Commons before the dissolution of Parliament . ^
2 Minority government before dissolution . ^ 3 Tory majority in the Commons before dissolution . ^ 4 Majority Tory government before dissolution . ^ 5 Majority
Liberal government before dissolution . ^ 6 Coalition government during the course of an undissolved hung parliament . ^ 7 Majority Conservative government
before dissolution . See
</item 1>

<item 2>
List of Prime Ministers of the United Kingdom
142 . ^ Jump up to : Eccleshall & Walker ( 2002 ) , p. 151 . ^ Jump up to : Eccleshall & Walker ( 2002 ) , p. 161 . Jump up ^ Eccleshall & Walker ( 2002 ) , p.
167 . ^ Jump up to : Eccleshall & Walker ( 2002 ) , p. 174 . ^ Jump up to : Eccleshall & Walker ( 2002 ) , p. 183 . ^ Jump up to : Eccleshall & Walker ( 2002 )
, p. 196 . ^ Jump up to : Eccleshall & Walker ( 2002 ) , p. 213 . Jump up ^ Eccleshall & Walker ( 2002 ) , p. 222 . Jump up ^ Eccleshall & Walker ( 2002 ) , p.
231 . Jump up ^ Eccleshall & Walker ( 2002 ) , p. 239 . Jump up ^ Eccleshall & Walker ( 2002 ) , p. 244 .
</item 2>

<item 3>
List of Prime Ministers of the United Kingdom
14 . Jump up ^ Eccleshall & Walker ( 2002 ) , p. 17 . ^ Jump up to : Eccleshall & Walker ( 2002 ) , p. 28 . Jump up ^ Eccleshall & Walker ( 2002 ) , p. 34 .
Jump up ^ Eccleshall & Walker ( 2002 ) , p. 36 . Jump up ^ Eccleshall & Walker ( 2002 ) , p. 42 . ^ Jump up to : Eccleshall & Walker ( 2002 ) , p. 46 . Jump up
^ Eccleshall & Walker ( 2002 ) , p. 54 . Jump up ^ Eccleshall & Walker ( 2002 ) , p. 61 . Jump up ^ Eccleshall & Walker ( 2002 ) , p. 64 . Jump up ^ Eccleshall
& Walker ( 2002 ) , p. 73 . ^ Jump up to : Eccleshall & Walker ( 2002 ) , p. 77 .
</item 3>

<item 4>
List of Prime Ministers of the United Kingdom
BBC News . 13 July 2016 . Retrieved 26 July 2016 . Lay summary ( 13 July 2016 ) . Bibliography ( edit ) Burt , Llewellyn C. ( 1874 ) . A synoptical history of
England ( 2nd ed . ) . London : Lockwood . Carpenter , Clive , ed. ( 1992 ) . The Guinness UK Data Book . Guinness . ISBN 978 - 0 - 85112 - 522 - 0 . Chapman ,
Richard A. ( 31 January 2002 ) . `` History : from earliest times to the present day '' . The Treasury in Public Policy - Making . Routledge . ISBN 978 - 1 -
134 - 86426 - 3 . Clarke , John ( 1999 ) . `` House of Hanover '' . In Fraser , Antonia . The Lives of the Kings & Queens of England . University of California
Press ( published February 2000 ) .
</item 4>

<item 5>
List of Prime Ministers of the United Kingdom
77 . ^ Jump up to : Eccleshall & Walker ( 2002 ) , p. 85 . Jump up ^ Eccleshall & Walker ( 2002 ) , p. 94 . Jump up ^ Eccleshall & Walker ( 2002 ) , p. 98 .
Jump up ^ Eccleshall & Walker ( 2002 ) , p. 101 . Jump up ^ Eccleshall & Walker ( 2002 ) , p. 106 . Jump up ^ Eccleshall & Walker ( 2002 ) , p. 116 . Jump up ^
Eccleshall & Walker ( 2002 ) , p. 120 . ^ Jump up to : Eccleshall & Walker ( 2002 ) , p. 123 . Jump up ^ Eccleshall & Walker ( 2002 ) , p. 128 . ^ Jump up to :
Eccleshall & Walker ( 2002 ) , p. 136 . ^ Jump up to : Eccleshall & Walker ( 2002 ) , p. 142 .
</item 5>

<item 6>
List of Prime Ministers of the United Kingdom
Jump up ^ Eccleshall & Walker ( 2002 ) , p. 244 . Jump up ^ Eccleshall & Walker ( 2002 ) , p. 252 . Jump up ^ Eccleshall & Walker ( 2002 ) , p. 262 . ^ Jump up
to : Eccleshall & Walker ( 2002 ) , p. 273 . ^ Jump up to : Eccleshall & Walker ( 2002 ) , p. 281 . Jump up ^ Eccleshall & Walker ( 2002 ) , p. 289 . ^ Jump up
to : Eccleshall & Walker ( 2002 ) , p. 295 . Jump up ^ Eccleshall & Walker ( 2002 ) , p. 305 . Jump up ^ Eccleshall & Walker ( 2002 ) , p. 315 . Jump up ^
Eccleshall & Walker ( 2002 ) , p. 320 . Jump up ^ Eccleshall & Walker ( 2002 ) , p. 329 . ^ Jump up to : Eccleshall & Walker ( 2002 ) , p.
</item 6>

<item 7>
List of Prime Ministers of the United Kingdom
V Edward VIII George VI Elizabeth II Whig ( 17 ) Tory ( 11 ) Conservative ( 17 ) Peelite ( 1 ) Liberal ( 7 ) Labour ( 6 ) National Labour ( 1 ) Portrait Name (
Birth -- Death ) Term of office Mandate Ministerial portfolios held as Prime Minister Party Ministry Monarch ( Reign ) Duration The Right Honourable Robert
Walpole 1st Earl of Orford KG KB PC MP for King 's Lynn ( 1676 -- 1745 ) 4 April 1721 11 February 1742 1722 First Lord of the Treasury Chancellor of the
Exchequer Leader of the House of Commons Whig Walpole -- Townshend George I ( 1714 -- 1727 ) 1727 George II ( 1727 -- 1760 ) 1734 Walpole 1741
7003761900000000000 ♠ 20 years and 314 days The Right Honourable Spencer Compton 1st Earl of Wilmington KG KB PC ( 1673 -- 1743 ) 16 February 1742 2 July 1743
-- First Lord of the Treasury Whig Carteret
</item 7>

<item 8>
List of Prime Ministers of the United Kingdom
-- N. Lab. -- et al . ) Edward VIII ( 1936 ) George VI ( 1936 -- 1952 ) 7002722000000000000 ♠ 1 year and 356 days The Right Honourable Neville Chamberlain FRS
MP for Birmingham Edgbaston ( 1869 -- 1940 ) 28 May 1937 10 May 1940 -- First Lord of the Treasury Leader of the House of Commons Conservative National IV ​
Chamberlain War 7003107900000000000 ♠ 2 years and 349 days The Right Honourable Winston Churchill CH TD PCc FRS MP for Epping ( 1874 -- 1965 ) 10 May 1940 26
July 1945 -- First Lord of the Treasury Minister of Defence Leader of the House of Commons Conservative Churchill War ( All parties ) ​ Churchill Caretaker (
Con. -- Lib. N . ) 7003190400000000000 ♠ 5 years and 78 days The Right Honourable Clement Attlee CH FRS MP for Limehouse -- Walthamstow West ( 1883 -- 1967 ) 26
July
</item 8>

<item 9>
List of Prime Ministers of the United Kingdom
-- 1881 ) See also § Premierships : 1 First Lord of the Treasury Leader of the House of Commons Conservative 27 February 1868 1 December 1868 --
7002279000000000000 ♠ 279 days The Right Honourable William Ewart Gladstone FSS MP for Greenwich ( 1809 -- 1898 ) See also § Premierships : 2 First Lord of the
Treasury Leader of the House of Commons Chancellor of the Exchequer Liberal Gladstone I 3 December 1868 17 February 1874 1868 7003190300000000000 ♠ 5 years and
77 days The Right Honourable Benjamin Disraeli 1st Earl of Beaconsfield KG PC FRS MP for Buckinghamshire ( 1804 -- 1881 ) See also § Premierships : 1 First Lord
of the Treasury Leader of the House of Commons Lord Keeper of the Privy Seal Leader of the House of Lords Conservative Disraeli II 20 February 1874 21 April
1880 1874 7003225300000000000 ♠ 6 years and 62 days The Right Honourable William
</item 9>

<item 10>
List of Prime Ministers of the United Kingdom
After the succession of King George I in 1714 , the arrangement of a commission of Lords of the Treasury as opposed to a single Lord High Treasurer became
permanent . For the next three years , the government was headed by Lord Townshend , who was appointed Secretary of State for the Northern Department .
Subsequently , Lord Stanhope and Lord Sunderland ran the government jointly , with Stanhope managing foreign affairs and Sunderland managing domestic affairs .
Stanhope died in February 1721 and Sunderland resigned two months later ; Townshend and Robert Walpole were then invited to form the next government . From that
point , the holder of the office of First Lord also unofficially held the status of Prime Minister . It was not until the Edwardian era that the title Prime
Minister was constitutionally recognised . With only few exceptions , the Prime Minister still holds the office of First Lord by constitutional convention .
Since 1721 ( edit ) By century and monarch 18th to 19th century George I George II George III George IV William IV Victoria 20th to 21st century Edward VII
George V Edward VIII George VI Elizabeth II Whig ( 17 ) Tory ( 11 ) Conservative (
</item 10>
=== END ITEM id=5a6386d78cfa5bb4 ===

=== ITEM id=580109cada46f566 ===
Gold sentences and facts:
Sentence 0: This Academy Awards ceremony is notable for Peter Finch becoming the first posthumous winner of an Oscar for acting , a feat matched only by fellow Australian Heath Ledger 32 years later ; Finch had suffered a fatal heart attack in mid-January . Beatrice Straight set another record by becoming the actor with the shortest performance ever in a film to win an acting Oscar , with only five minutes and two seconds of screen - time in Network . Network , along with All the President 's Men , were the two biggest champs of the ceremony with four Oscars each , but Best Picture and Best Director ultimately went to Rocky .
  Fact 0.0: This Academy Awards ceremony is notable for Peter Finch becoming the first posthumous winner of an Oscar for acting.
  Fact 0.1: Peter Finch's posthumous acting Oscar was matched only by fellow Australian Heath Ledger, 32 years later.
  Fact 0.2: Peter Finch had suffered a fatal heart attack in mid-January.
  Fact 0.3: Beatrice Straight set a record as the actor with the shortest performance ever in a film to win an acting Oscar.
  Fact 0.4: Beatrice Straight's performance in Network had only five minutes and two seconds of screen time.
  Fact 0.5: Beatrice Straight won an acting Oscar for her performance in Network.
  Fact 0.6: Network and All the President's Men were the two biggest winners of the ceremony.
  Fact 0.7: Network won four Oscars.
  Fact 0.8: All the President's Men won four Oscars.
  Fact 0.9: Best Picture and Best Director ultimately went to Rocky.

Retrieved items:
<item 1>
49th Academy Awards
Retrieved 2011 - 10 - 03 . Retrieved from `` https://en.wikipedia.org/w/index.php?title=49th_Academy_Awards&oldid=834282163 '' Categories : Academy Awards
ceremonies 1976 film awards 1977 in Los Angeles 1977 in American cinema March 1977 events 1976 awards in the United States Hidden categories : Articles with
hCards Talk Contents About Wikipedia Afrikaans Asturianu Български Català Deutsch Ελληνικά Español فارسی Français Galego Bahasa Indonesia Italiano ქართული
Magyar Nederlands 日本 語 Polski Português Русский Svenska ไทย 中文 14 more Edit links This page was last edited on 4 April 2018 , at 21 : 32 . Text is available
under the Creative Commons Attribution - ShareAlike License ; additional terms may apply . By using this site , you agree to the Terms of Use and Privacy Policy
. Wikipedia ® is a registered trademark of the Wikimedia Foundation , Inc.
</item 1>

<item 2>
Gerald Ford
Buckley Ronald Reagan Harold Stassen Third party and independent candidates American Party Nominee Thomas J. Anderson American Independent Party Nominee Lester
Maddox Communist Party Nominee Gus Hall VP nominee Jarvis Tyner Libertarian Party Nominee Roger MacBride VP nominee David Bergland People 's Party Nominee
Margaret Wright VP nominee Benjamin Spock Prohibition Party Nominee Ben Bubar VP nominee Earl Dodge Socialist Workers Party Nominee Peter Camejo VP nominee
Willie Mae Reid U.S. Labor Party Nominee Lyndon LaRouche Other 1976 elections : House Senate Gubernatorial Theodore Roosevelt Award winners 1967 : Eisenhower
1968 : Saltonstall 1969 : White 1970 : Hovde 1971 : Kraft Jr . 1972 : Holland 1973 : Bradley 1974 : Owens 1975 : Ford 1976 : Hamilton 1977 : Bradley 1978 :
Zornow 1979 : Chandler 1980 : Cooley 1981 : Linkletter 1982 : Cosby 1983 : Palmer 1984 : Lawrence 1985 : Fleming 1986 : Bush 1987 : Zable 1988 : Not presented
1989 : Ebert 1990 :
</item 2>

<item 3>
49th Academy Awards
producers John G. Avildsen -- Rocky Alan J. Pakula -- All the President 's Men Ingmar Bergman -- Face to Face Sidney Lumet -- Network Lina Wertmüller -- Seven
Beauties Best Actor Best Actress Peter Finch -- Network as Howard Beale Robert De Niro -- Taxi Driver as Travis Bickle Giancarlo Giannini -- Seven Beauties as
Pasqualino William Holden -- Network as Max Schumacher Sylvester Stallone -- Rocky as Rocky Balboa Faye Dunaway -- Network as Diana Christensen Marie -
Christine Barrault -- Cousin Cousine as Marthe Talia Shire -- Rocky as Adriana Pennino Sissy Spacek -- Carrie as Carrie White Liv Ullmann -- Face to Face as
Jenny Isaksson Best Supporting Actor Best Supporting Actress Jason Robards -- All the President 's Men as Ben Bradlee Ned Beatty -- Network as Arthur Jensen
Burgess Meredith -- Rocky as `` Mickey '' Goldmill Laurence Olivier -- Marathon Man as Dr.
</item 3>

<item 4>
49th Academy Awards
Mitchell , Richard Tyler and Douglas Williams A Star Is Born -- Robert Glass , Robert Knudson , Tom Overton and Dan Wallin Best Foreign Language Film Best
Costume Design Black and White in Color ( Côte d'Ivoire ) Cousin Cousine ( France ) Jacob the Liar ( East Germany ) Nights and Days ( Poland ) Seven Beauties (
Italy ) Fellini 's Casanova -- Danilo Donati Bound for Glory -- William Ware Theiss The Incredible Sarah -- Anthony Mendleson The Passover Plot -- Mary Wills
The Seven - Per - Cent Solution -- Alan Barrett Best Art Direction Best Cinematography All the President 's Men -- Art Direction : George Jenkins ; Set
Decoration : George Gaines The Incredible Sarah -- Art Direction : Elliot Scott ; Set Decoration : Norman Reynolds The Last Tycoon -- Art Direction : Gene
Callahan and Jack T. Collis ; Set Decoration : Jerry Wunderlich Logan 's Run -- Art Direction : Dale Hennesy ; Set Decoration : Robert De Vestel The Shootist --
Art Direction : Robert F. Boyle ; Set Decoration : Arthur Jeph Parker Bound for Glory -- Haskell Wexler King Kong -- Richard H.
</item 4>

<item 5>
The Deer Hunter
( 1968 ) Midnight Cowboy ( 1969 ) Patton ( 1970 ) The French Connection ( 1971 ) The Godfather ( 1972 ) The Sting ( 1973 ) The Godfather Part II ( 1974 ) One
Flew Over the Cuckoo 's Nest ( 1975 ) 1976 -- 2000 Rocky ( 1976 ) Annie Hall ( 1977 ) The Deer Hunter ( 1978 ) Kramer vs. Kramer ( 1979 ) Ordinary People ( 1980
) Chariots of Fire ( 1981 ) Gandhi ( 1982 ) Terms of Endearment ( 1983 ) Amadeus ( 1984 ) Out of Africa ( 1985 ) Platoon ( 1986 ) The Last Emperor ( 1987 ) Rain
Man ( 1988 ) Driving Miss Daisy ( 1989 ) Dances with Wolves ( 1990 ) The Silence of the Lambs ( 1991 ) Unforgiven ( 1992 ) Schindler 's List ( 1993 ) Forrest
Gump ( 1994 ) Braveheart ( 1995 ) The English Patient ( 1996 ) Titanic (
</item 5>

<item 6>
The Deer Hunter
and most controversial , major theatrical films to be critical of the American involvement in Vietnam following 1975 when the war officially ended . While the
film opened the same year as Hal Ashby 's Coming Home , Sidney Furie 's The Boys in Company C , and Ted Post 's Go Tell the Spartans , it was the first film
about Vietnam to reach a wide audience and critical acclaim , culminating in the winning of the Oscar for Best Picture .
</item 6>

<item 7>
49th Academy Awards
Christian Szell Burt Young -- Rocky as Paulie Pennino Beatrice Straight -- Network as Louise Schumacher Jane Alexander -- All the President 's Men as Judy
Graham Hoback Jodie Foster -- Taxi Driver as Iris Steensma Lee Grant -- Voyage of the Damned as Lillian Rosen Piper Laurie -- Carrie as Margaret White Best
Screenplay Written Directly for the Screen Best Screenplay Based on Material from Another Medium Network -- Paddy Chayefsky Cousin Cousine -- Jean - Charles
Tacchella and Danièle Thompson The Front -- Walter Bernstein Rocky -- Sylvester Stallone Seven Beauties -- Lina Wertmüller All the President 's Men -- William
Goldman based on the book by Carl Bernstein and Bob Woodward Bound for Glory -- Robert Getchell based on the book by Woody Guthrie Fellini 's Casanova --
Federico Fellini and Bernardino Zapponi based on the autobiography Histoire de ma vie by Giacomo Casanova The Seven - Per - Cent Solution -- Nicholas Meyer
based on his novel Voyage of the Damned -- David Butler and Steve Shagan based on the book by Gordon Thomas and Max Morgan Witts Best Documentary Feature Best
Documentary Short Subject Harlan County , USA -- Barbara
</item 7>

<item 8>
49th Academy Awards
USA -- Barbara Kopple Hollywood on Trial -- David Helpern Off the Edge -- Michael Firth People of the Wind -- Anthony Howarth and David Koff Volcano : An
Inquiry into the Life and Death of Malcolm Lowry -- Donald Brittain and John Kramer Number Our Days -- Lynne Littman and Barbara Myerhoff American Shoeshine
Blackwood The End of the Road Universe Best Live Action Short Film Best Animated Short Film In the Region of Ice -- Peter Werner Kudzu -- Marjorie Anne Short
The Morning Spider -- Julian Chagrin and Claude Chagrin Nightlife -- Claire Wilbur and Robin Lehman Number One -- Dyan Cannon and Vince Cannon Leisure --
Suzanne Baker Dedalo -- Manfredo Manfredi The Street -- Caroline Leaf and Guy Glover Best Original Score Best Original Song Score or Adaptation Score The Omen
-- Jerry Goldsmith Obsession -- Bernard Herrmann ( posthumous nomination ) The Outlaw Josey Wales -- Jerry Fielding Taxi Driver -- Bernard Herrmann ( posthumous
nomination ) Voyage of the Damned -- Lalo Schifrin Bound for Glory -- Leonard Rosenman Bugsy Malone -- Paul Williams A Star Is Born -- Roger Kellaway Best
Original Song Best Sound ``
</item 8>

<item 9>
49th Academy Awards
The two events conflicted again five years later in 1982 . Contents ( hide ) 1 Winners and nominees 1.1 Special Achievement Award 1.2 Irving G. Thalberg
Memorial Award 1.3 Multiple nominations and awards 2 Presenters and performers 2.1 Presenters 2.2 Performers 3 See also 4 References Winners and Nominees ( edit
) Peter Finch , Best Actor winner Faye Dunaway , Best Actress winner Jason Robards , Best Supporting Actor winner Beatrice Straight , Best Supporting Actress
winner Winners are listed first , highlighted in boldface and indicated with a double dagger ( ) . Best Picture Best Director Rocky -- Robert Chartoff and Irwin
Winkler , producers All the President 's Men -- Walter Coblenz , producer Bound for Glory -- Robert F. Blumofe and Harold Leventhal , producers Network --
Howard Gottfried , producer Taxi Driver -- Julia Phillips and Michael Phillips , producers John G. Avildsen -- Rocky Alan J.
</item 9>

<item 10>
Summer of '42
On this point , Raucher said in May 2002 that his ten percent of the gross , in addition to royalties from book sales , `` has paid bills ever since . '' As
well as being a commercial success , Summer of ' 42 also received rave critical reviews . It went on to be nominated for over a dozen awards , including Golden
Globe Awards for `` Best Motion Picture -- Drama '' and `` Best Director '' , and five Academy Award nominations for Best Original Music Score , Best
Cinematography , Best Editing , Best Writing - Story and Best Screenplay . Ultimately , the film won two awards : the 1972 44th Academy Awards Oscar for
Original Dramatic Score , and the 1971 BAFTA Anthony Asquith Award for Film Music , both to Michel Legrand . It counted among its fans Stanley Kubrick , who had
the film play on a television in a scene in The Shining .
</item 10>
=== END ITEM id=580109cada46f566 ===

=== ITEM id=9119f762f458cbb4 ===
Gold sentences and facts:
Sentence 0: Country Name of animal Scientific name Pictures Ref . Algeria Fennec fox Vulpes zerda Angola Red - crested turaco ( national bird ) Tauraco erythrolophus Anguilla Zenaida dove Zenaida aurita Antigua and Barbuda Fallow deer ( national animal ) Dama dama Frigate ( national bird ) Fregata magnificens Hawksbill turtle ( national sea creature ) Eretmochelys imbricata Argentina Rufous hornero Furnarius rufus Australia Red kangaroo ( national animal ) Macropus rufus Emu ( national bird ) Dromaius novaehollandiae Austria Black eagle Ictinaetus malaiensis Azerbaijan Karabakh horse Equus ferus caballus Bangladesh Royal Bengal tiger ( national animal ) Panthera tigris tigris Magpie robin ( national bird ) Copsychus saularis Ilish ( national fish ) Tenualosa ilisha Belarus European bison Bison bonasus Belgium Lion ( heraldic Leo Belgicus ) Panthera leo Belize Baird 's tapir ( national animal ) Tapirus bairdii Keel - billed toucan ( national bird ) Ramphastos sulfuratus Bhutan Druk Mythical Takin Budorcas taxicolor Brazil Rufous - bellied thrush Turdus rufiventris Cambodia Kouprey Bos sauveli Canada North American beaver ( sovereignty animal symbol ) Castor canadensis Canadian horse ( national horse ) Equus ferus caballus China Giant panda ( national animal ) Ailuropoda melanoleuca Chinese dragon ( national animal ) Mythical Red - crowned crane ( national bird ) Grus japonensis Democratic Republic of the Congo Okapi Okapia johnstoni Colombia Andean condor Vultur gryphus Costa Rica Yigüirro ( national bird ) Turdus grayi White - tailed deer ( national animal ) Odocoileus virginianus West Indian manatee ( national aquatic animal ) Trichechus manatus Croatia Pine marten Martes martes Cuba Cuban trogon Priotelus temnurus Cyprus Cypriot mouflon Ovis orientalis Czech Republic Double - tailed lion Mythical Denmark Mute swan ( national bird ) Cygnus olor Small tortoiseshell ( national butterfly ) Aglais urticae Egypt Steppe eagle Aquila nipalensis Estonia Barn swallow ( national bird ) Hirundo rustica Eritrea Arabian camel Camelus dromedarius Ethiopia Lion Panthera leo Faroe Islands Eurasian oystercatcher ( national bird ) Haematopus ostralgus Finland Brown bear ( national animal ) Ursus arctos Whooper swan ( national bird ) Cygnus cygnus Holly blue ( national butterfly ) Celastrina argiolus European perch ( national fish ) Perca fluviatilis Seven - spot ladybird ( national insect ) Coccinella septempunctata France Gallic rooster Gallus gallus domesticus Germany Unspecified Eagle Members of Accipitridae Greece Phoenix Mythical Dolphins Delphinus Guatemala Quetzal Pharomachrus mocinno Haiti Hispaniolan Trogan Priotelus roseigaster Honduras White - tailed deer Odocoileus virginianus Hungary Turul Mythical ( most probably Falco cherrug ) Iceland Gyrfalcon Falco rusticolus India Bengal tiger ( National predator ) Panthera tigris tigris Indian peafowl ( National bird ) Pavo cristatus King cobra ( National reptile ) Ophiophagus hannah Ganges river dolphin ( National aquatic animal ) Platanista gangetica gangetica Indian elephant ( National heritage animal ) Elephas maximus indicus Indonesia Komodo dragon ( national animal ) Varanus komodoensis Garuda ( historic national animal ) Mythical Iran Persian lion Panthera leo persica Iraq Chukar partridge Alectoris chukar Goat ( KRG ) Capra aegagrus Israel Hoopoe ( national bird ) Upupa epops Jamaica Red - billed streamertail Trochilus polytmus Japan Green pheasant ( national bird ) Phasianus versicolor Carp ( national fish ) Cyprinus carpio Latvia White wagtail ( national bird ) Motacilla alba Two - spotted ladybird ( national insect ) Adalia bipunctata Lithuania White stork Ciconia ciconia Macedonia Lion ( in Macedonian heraldry ) Panthera leo Madagascar Zebu ( in Malagasy heraldry ) Bos primigenius indicus Ring - tailed lemur ( national animal ) Lemur catta Malaysia Malayan tiger ( national animal ) Panthera tigris jacksoni Malta Pharaoh Hound ( national animal ) Canis lupus familiaris ( 1 ) Moldova Aurochs ( national animal ) Bos primigenius Mauritius Dodo Raphus cucullatus Mexico Golden eagle ( national animal ) Aquila chrysaetos Xoloitzcuintli ( national dog ) Canis lupus familiaris Jaguar ( national mammal ) Panthera onca Grasshopper ( national arthropod ) Sphenarium purpurascens Vaquita ( national marine mammal ) Phocoena sinus Nepal Cow ( national animal ) Bos taurus indicus Danphe ( national bird ) Lophophorus impejanus Netherlands Lion ( heraldic symbol ) Panthera leo Black - tailed godwit ( national bird ) Limosa limosa New Zealand Kiwi Apteryx sp . North Korea Chollima Mythical Norway Lion ( royal national animal ) Panthera leo White - throated dipper ( national bird ) Cinclus cinclus Fjord horse ( national horse ) Equus ferus caballus Oman Arabian oryx ( national animal ) Oryx leucoryx Pakistan Markhor ( National animal ) Capra falconeri Indus river dolphin ( National aquatic mammal ) Platanista indicus minor minor Chukar ( National bird ) Alectoris chukar Indus crocodile ( National reptile ) Crocodylus palustris Shaheen falcon ( Heritage bird ) Falco peregrinus peregrinator Snow leopard ( National predator ) Panthera uncia Panama Harpy eagle Harpia harpyja Papua New Guinea Dugong ( national marine mammal ) Dugong dugon Peru Vicuña ( national animal ) Vicugna vicugna Philippines Carabao ( national animal ) Bubalus bubalis Philippine eagle ( national bird ) Pithecophaga jefferyi Poland White - tailed eagle ( national bird ) Haliaeetus albicilla White stork ( national bird ) Ciconia ciconia European bison ( national animal ) Bison bonasus Qatar Arabian oryx Oryx leucoryx Romania Lynx Lynx lynx Russia Eurasian brown bear Ursus arctos arctos Rwanda Leopard Panthera pardus Saint Vincent and the Grenadines Saint Vincent amazon Amazona guildingii Serbia Wolf Canis lupus lupus White eagle Mythical Lynx Lynx lynx Somalia Leopard ( national animal ) Panthera pardus South Africa Springbok ( national animal ) Antidorcas marsupialis Blue crane ( national bird ) Anthropoides paradiseus Galjoen ( national fish ) Dichistius capensis South Korea Siberian tiger Panthera tigris altaica Korean magpie ( national bird ) Pica ( pica ) sericea Sri Lanka Sri Lankan junglefowl ( national bird ) Gallus lafayettii Tanzania Giraffe Giraffa camelopardalis Thailand Elephant Elephas maximus United Arab Emirates Arabian oryx ( national animal ) Oryx leucoryx Peregrine falcon ( national bird ) Falco peregrinus United Kingdom Lion ( United Kingdom , England , Scotland , Wales ) Panthera leo Bulldog ( Britain as `` British Bulldog '' ) Canis lupus familiaris Red kite ( Wales ) Milvus milvus Unicorn ( Scotland ) Mythical Y Ddraig Goch ( Welsh Dragon Wales ) Mythical United States Bald eagle ( national bird ) Haliaeetus leucocephalus American bison ( national mammal ) Bison bison
  Fact 0.0: In Algeria, the Fennec fox (Vulpes zerda) is listed as the animal.
  Fact 0.1: In Angola, the Red-crested turaco (Tauraco erythrolophus) is listed as the national bird.
  Fact 0.2: In Anguilla, the Zenaida dove (Zenaida aurita) is listed.
  Fact 0.3: In Antigua and Barbuda, the Fallow deer (Dama dama) is listed as the national animal.
  Fact 0.4: In Antigua and Barbuda, the Frigate (Fregata magnificens) is listed as the national bird.
  Fact 0.5: In Antigua and Barbuda, the Hawksbill turtle (Eretmochelys imbricata) is listed as the national sea creature.
  Fact 0.6: In Argentina, the Rufous hornero (Furnarius rufus) is listed.
  Fact 0.7: In Australia, the Red kangaroo (Macropus rufus) is listed as the national animal.
  Fact 0.8: In Australia, the Emu (Dromaius novaehollandiae) is listed as the national bird.
  Fact 0.9: In Austria, the Black eagle (Ictinaetus malaiensis) is listed.
  Fact 0.10: In Azerbaijan, the Karabakh horse (Equus ferus caballus) is listed.
  Fact 0.11: In Bangladesh, the Royal Bengal tiger (Panthera tigris tigris) is listed as the national animal.
  Fact 0.12: In Bangladesh, the Magpie robin (Copsychus saularis) is listed as the national bird.
  Fact 0.13: In Bangladesh, the Ilish (Tenualosa ilisha) is listed as the national fish.
  Fact 0.14: In Belarus, the European bison (Bison bonasus) is listed.
  Fact 0.15: In Belgium, the Lion (Panthera leo) is listed as the heraldic Leo Belgicus.
  Fact 0.16: In Belize, Baird's tapir (Tapirus bairdii) is listed as the national animal.
  Fact 0.17: In Belize, the Keel-billed toucan (Ramphastos sulfuratus) is listed as the national bird.
  Fact 0.18: In Bhutan, the Druk is listed as a mythical animal.
  Fact 0.19: In Bhutan, the Takin (Budorcas taxicolor) is listed.
  Fact 0.20: In Brazil, the Rufous-bellied thrush (Turdus rufiventris) is listed.
  Fact 0.21: In Cambodia, the Kouprey (Bos sauveli) is listed.
  Fact 0.22: In Canada, the North American beaver (Castor canadensis) is listed as the sovereignty animal symbol.
  Fact 0.23: In Canada, the Canadian horse (Equus ferus caballus) is listed as the national horse.
  Fact 0.24: In China, the Giant panda (Ailuropoda melanoleuca) is listed as the national animal.
  Fact 0.25: In China, the Chinese dragon is listed as a mythical national animal.
  Fact 0.26: In China, the Red-crowned crane (Grus japonensis) is listed as the national bird.
  Fact 0.27: In the Democratic Republic of the Congo, the Okapi (Okapia johnstoni) is listed.
  Fact 0.28: In Colombia, the Andean condor (Vultur gryphus) is listed.
  Fact 0.29: In Costa Rica, the Yigüirro (Turdus grayi) is listed as the national bird.
  Fact 0.30: In Costa Rica, the White-tailed deer (Odocoileus virginianus) is listed as the national animal.
  Fact 0.31: In Costa Rica, the West Indian manatee (Trichechus manatus) is listed as the national aquatic animal.
  Fact 0.32: In Croatia, the Pine marten (Martes martes) is listed.
  Fact 0.33: In Cuba, the Cuban trogon (Priotelus temnurus) is listed.
  Fact 0.34: In Cyprus, the Cypriot mouflon (Ovis orientalis) is listed.
  Fact 0.35: In the Czech Republic, the Double-tailed lion is listed as a mythical animal.
  Fact 0.36: In Denmark, the Mute swan (Cygnus olor) is listed as the national bird.
  Fact 0.37: In Denmark, the Small tortoiseshell (Aglais urticae) is listed as the national butterfly.
  Fact 0.38: In Egypt, the Steppe eagle (Aquila nipalensis) is listed.
  Fact 0.39: In Estonia, the Barn swallow (Hirundo rustica) is listed as the national bird.
  Fact 0.40: In Eritrea, the Arabian camel (Camelus dromedarius) is listed.
  Fact 0.41: In Ethiopia, the Lion (Panthera leo) is listed.
  Fact 0.42: In the Faroe Islands, the Eurasian oystercatcher (Haematopus ostralgus) is listed as the national bird.
  Fact 0.43: In Finland, the Brown bear (Ursus arctos) is listed as the national animal.
  Fact 0.44: In Finland, the Whooper swan (Cygnus cygnus) is listed as the national bird.
  Fact 0.45: In Finland, the Holly blue (Celastrina argiolus) is listed as the national butterfly.
  Fact 0.46: In Finland, the European perch (Perca fluviatilis) is listed as the national fish.
  Fact 0.47: In Finland, the Seven-spot ladybird (Coccinella septempunctata) is listed as the national insect.
  Fact 0.48: In France, the Gallic rooster (Gallus gallus domesticus) is listed.
  Fact 0.49: In Germany, the Unspecified Eagle (a member of Accipitridae) is listed.
  Fact 0.50: In Greece, the Phoenix is listed as a mythical animal.
  Fact 0.51: In Greece, the Dolphins (Delphinus) are listed.
  Fact 0.52: In Guatemala, the Quetzal (Pharomachrus mocinno) is listed.
  Fact 0.53: In Haiti, the Hispaniolan Trogan (Priotelus roseigaster) is listed.
  Fact 0.54: In Honduras, the White-tailed deer (Odocoileus virginianus) is listed.
  Fact 0.55: In Hungary, the Turul is listed as a mythical animal, most probably Falco cherrug.
  Fact 0.56: In Iceland, the Gyrfalcon (Falco rusticolus) is listed.
  Fact 0.57: In India, the Bengal tiger (Panthera tigris tigris) is listed as the National predator.
  Fact 0.58: In India, the Indian peafowl (Pavo cristatus) is listed as the National bird.
  Fact 0.59: In India, the King cobra (Ophiophagus hannah) is listed as the National reptile.
  Fact 0.60: In India, the Ganges river dolphin (Platanista gangetica gangetica) is listed as the National aquatic animal.
  Fact 0.61: In India, the Indian elephant (Elephas maximus indicus) is listed as the National heritage animal.
  Fact 0.62: In Indonesia, the Komodo dragon (Varanus komodoensis) is listed as the national animal.
  Fact 0.63: In Indonesia, the Garuda is listed as the historic national animal, a mythical one.
  Fact 0.64: In Iran, the Persian lion (Panthera leo persica) is listed.
  Fact 0.65: In Iraq, the Chukar partridge (Alectoris chukar) is listed.
  Fact 0.66: In Iraq, the Goat (Capra aegagrus) is listed for KRG.
  Fact 0.67: In Israel, the Hoopoe (Upupa epops) is listed as the national bird.
  Fact 0.68: In Jamaica, the Red-billed streamertail (Trochilus polytmus) is listed.
  Fact 0.69: In Japan, the Green pheasant (Phasianus versicolor) is listed as the national bird.
  Fact 0.70: In Japan, the Carp (Cyprinus carpio) is listed as the national fish.
  Fact 0.71: In Latvia, the White wagtail (Motacilla alba) is listed as the national bird.
  Fact 0.72: In Latvia, the Two-spotted ladybird (Adalia bipunctata) is listed as the national insect.
  Fact 0.73: In Lithuania, the White stork (Ciconia ciconia) is listed.
  Fact 0.74: In Macedonia, the Lion (Panthera leo) is listed as the animal in Macedonian heraldry.
  Fact 0.75: In Madagascar, the Zebu (Bos primigenius indicus) is listed as the animal in Malagasy heraldry.
  Fact 0.76: In Madagascar, the Ring-tailed lemur (Lemur catta) is listed as the national animal.
  Fact 0.77: In Malaysia, the Malayan tiger (Panthera tigris jacksoni) is listed as the national animal.
  Fact 0.78: In Malta, the Pharaoh Hound (Canis lupus familiaris) is listed as the national animal, with reference 1.
  Fact 0.79: In Moldova, the Aurochs (Bos primigenius) is listed as the national animal.
  Fact 0.80: In Mauritius, the Dodo (Raphus cucullatus) is listed.
  Fact 0.81: In Mexico, the Golden eagle (Aquila chrysaetos) is listed as the national animal.
  Fact 0.82: In Mexico, the Xoloitzcuintli (Canis lupus familiaris) is listed as the national dog.
  Fact 0.83: In Mexico, the Jaguar (Panthera onca) is listed as the national mammal.
  Fact 0.84: In Mexico, the Grasshopper (Sphenarium purpurascens) is listed as the national arthropod.
  Fact 0.85: In Mexico, the Vaquita (Phocoena sinus) is listed as the national marine mammal.
  Fact 0.86: In Nepal, the Cow (Bos taurus indicus) is listed as the national animal.
  Fact 0.87: In Nepal, the Danphe (Lophophorus impejanus) is listed as the national bird.
  Fact 0.88: In the Netherlands, the Lion (Panthera leo) is listed as the heraldic symbol.
  Fact 0.89: In the Netherlands, the Black-tailed godwit (Limosa limosa) is listed as the national bird.
  Fact 0.90: In New Zealand, the Kiwi (Apteryx sp.) is listed.
  Fact 0.91: In North Korea, the Chollima is listed as a mythical animal.
  Fact 0.92: In Norway, the Lion (Panthera leo) is listed as the royal national animal.
  Fact 0.93: In Norway, the White-throated dipper (Cinclus cinclus) is listed as the national bird.
  Fact 0.94: In Norway, the Fjord horse (Equus ferus caballus) is listed as the national horse.
  Fact 0.95: In Oman, the Arabian oryx (Oryx leucoryx) is listed as the national animal.
  Fact 0.96: In Pakistan, the Markhor (Capra falconeri) is listed as the National animal.
  Fact 0.97: In Pakistan, the Indus river dolphin (Platanista indicus minor minor) is listed as the National aquatic mammal.
  Fact 0.98: In Pakistan, the Chukar (Alectoris chukar) is listed as the National bird.
  Fact 0.99: In Pakistan, the Indus crocodile (Crocodylus palustris) is listed as the National reptile.
  Fact 0.100: In Pakistan, the Shaheen falcon (Falco peregrinus peregrinator) is listed as the Heritage bird.
  Fact 0.101: In Pakistan, the Snow leopard (Panthera uncia) is listed as the National predator.
  Fact 0.102: In Panama, the Harpy eagle (Harpia harpyja) is listed.
  Fact 0.103: In Papua New Guinea, the Dugong (Dugong dugon) is listed as the national marine mammal.
  Fact 0.104: In Peru, the Vicuña (Vicugna vicugna) is listed as the national animal.
  Fact 0.105: In the Philippines, the Carabao (Bubalus bubalis) is listed as the national animal.
  Fact 0.106: In the Philippines, the Philippine eagle (Pithecophaga jefferyi) is listed as the national bird.
  Fact 0.107: In Poland, the White-tailed eagle (Haliaeetus albicilla) is listed as the national bird.
  Fact 0.108: In Poland, the White stork (Ciconia ciconia) is listed as the national bird.
  Fact 0.109: In Poland, the European bison (Bison bonasus) is listed as the national animal.
  Fact 0.110: In Qatar, the Arabian oryx (Oryx leucoryx) is listed.
  Fact 0.111: In Romania, the Lynx (Lynx lynx) is listed.
  Fact 0.112: In Russia, the Eurasian brown bear (Ursus arctos arctos) is listed.
  Fact 0.113: In Rwanda, the Leopard (Panthera pardus) is listed.
  Fact 0.114: In Saint Vincent and the Grenadines, the Saint Vincent amazon (Amazona guildingii) is listed.
  Fact 0.115: In Serbia, the Wolf (Canis lupus lupus) is listed.
  Fact 0.116: In Serbia, the White eagle is listed as a mythical animal.
  Fact 0.117: In Serbia, the Lynx (Lynx lynx) is listed.
  Fact 0.118: In Somalia, the Leopard (Panthera pardus) is listed as the national animal.
  Fact 0.119: In South Africa, the Springbok (Antidorcas marsupialis) is listed as the national animal.
  Fact 0.120: In South Africa, the Blue crane (Anthropoides paradiseus) is listed as the national bird.
  Fact 0.121: In South Africa, the Galjoen (Dichistius capensis) is listed as the national fish.
  Fact 0.122: In South Korea, the Siberian tiger (Panthera tigris altaica) is listed.
  Fact 0.123: In South Korea, the Korean magpie (Pica (pica) sericea) is listed as the national bird.
  Fact 0.124: In Sri Lanka, the Sri Lankan junglefowl (Gallus lafayettii) is listed as the national bird.
  Fact 0.125: In Tanzania, the Giraffe (Giraffa camelopardalis) is listed.
  Fact 0.126: In Thailand, the Elephant (Elephas maximus) is listed.
  Fact 0.127: In the United Arab Emirates, the Arabian oryx (Oryx leucoryx) is listed as the national animal.
  Fact 0.128: In the United Arab Emirates, the Peregrine falcon (Falco peregrinus) is listed as the national bird.
  Fact 0.129: In the United Kingdom, the Lion (Panthera leo) is listed for the United Kingdom, England, Scotland and Wales.
  Fact 0.130: In the United Kingdom, the Bulldog (Canis lupus familiaris) is listed as Britain, called the British Bulldog.
  Fact 0.131: In the United Kingdom, the Red kite (Milvus milvus) is listed for Wales.
  Fact 0.132: In the United Kingdom, the Unicorn is listed as a mythical animal for Scotland.
  Fact 0.133: In the United Kingdom, the Y Ddraig Goch (Welsh Dragon) is listed as a mythical animal for Wales.
  Fact 0.134: In the United States, the Bald eagle (Haliaeetus leucocephalus) is listed as the national bird.
  Fact 0.135: In the United States, the American bison (Bison bison) is listed as the national mammal.

Retrieved items:
<item 1>
List of national animals
Greece Phoenix Mythical Dolphins Delphinus Guatemala Quetzal Pharomachrus mocinno Haiti Hispaniolan Trogan Priotelus roseigaster Honduras White - tailed deer
Odocoileus virginianus Hungary Turul Mythical ( most probably Falco cherrug ) Iceland Gyrfalcon Falco rusticolus India Bengal tiger ( National predator )
Panthera tigris tigris Indian peafowl ( National bird ) Pavo cristatus King cobra ( National reptile ) Ophiophagus hannah Ganges river dolphin ( National
aquatic animal ) Platanista gangetica gangetica Indian elephant ( National heritage animal ) Elephas maximus indicus Indonesia Komodo dragon ( national animal )
Varanus komodoensis Garuda ( historic national animal ) Mythical Iran Persian lion Panthera leo persica Iraq Chukar partridge Alectoris chukar Goat ( KRG )
Capra aegagrus Israel Hoopoe ( national bird ) Upupa epops Jamaica Red - billed streamertail Trochilus polytmus Japan Green pheasant ( national bird ) Phasianus
versicolor Carp ( national fish ) Cyprinus carpio Latvia White wagtail (
</item 1>

<item 2>
List of national animals
Canadian Heritage . 2016 . Retrieved 1 December 2016 . Jump up ^ `` National Symbols of China '' . 123independenceday.com . Jump up ^ `` China National Symbols
'' . worldatlast.com. 2015 . Retrieved 2017 - 03 - 28 . Jump up ^ `` China Considers Red - crowned Crane for National Bird '' . www.china.org.cn . Jump up ^ ``
El cóndor , ave nacional - banrepcultural.org '' . Jump up ^ `` 16 de diciembre : Declaración del yigüirro como ave nacional de Costa Rica '' . Jump up ^ `` El
Venado Cola Blanca , Símbolo Nacional de la Fauna Silvestre '' . Jump up ^ `` Manatee Now Costa Rica 's New National Symbol '' . QCOSTARICA . Archived from the
original on 14 September 2014 . Jump up ^ Bousfield , Jonathan ( 2003 ) . Croatia . Rough Guides . p. 72 . ISBN 1843530848 .
</item 2>

<item 3>
List of national animals
Magpie robin ( national bird ) Copsychus saularis Ilish ( national fish ) Tenualosa ilisha Belarus European bison Bison bonasus Belgium Lion ( heraldic Leo
Belgicus ) Panthera leo Belize Baird 's tapir ( national animal ) Tapirus bairdii Keel - billed toucan ( national bird ) Ramphastos sulfuratus Bhutan Druk
Mythical Takin Budorcas taxicolor Brazil Rufous - bellied thrush Turdus rufiventris Cambodia Kouprey Bos sauveli Canada North American beaver ( sovereignty
animal symbol ) Castor canadensis Canadian horse ( national horse ) Equus ferus caballus China Giant panda ( national animal ) Ailuropoda melanoleuca Chinese
dragon ( national animal ) Mythical Red - crowned crane ( national bird ) Grus japonensis Democratic Republic of the Congo Okapi Okapia johnstoni Colombia
Andean condor Vultur gryphus Costa Rica Yigüirro ( national bird ) Turdus grayi White - tailed deer ( national animal ) Odocoileus virginianus West Indian
manatee ( national aquatic animal ) Trichechus manatus Croatia Pine
</item 3>

<item 4>
List of national animals
Vicugna vicugna Philippines Carabao ( national animal ) Bubalus bubalis Philippine eagle ( national bird ) Pithecophaga jefferyi Poland White - tailed eagle (
national bird ) Haliaeetus albicilla White stork ( national bird ) Ciconia ciconia European bison ( national animal ) Bison bonasus Qatar Arabian oryx Oryx
leucoryx Romania Lynx Lynx lynx Russia Eurasian brown bear Ursus arctos arctos Rwanda Leopard Panthera pardus Saint Vincent and the Grenadines Saint Vincent
amazon Amazona guildingii Serbia Wolf Canis lupus lupus White eagle Mythical Lynx Lynx lynx Somalia Leopard ( national animal ) Panthera pardus South Africa
Springbok ( national animal ) Antidorcas marsupialis Blue crane ( national bird ) Anthropoides paradiseus Galjoen ( national fish ) Dichistius capensis South
Korea Siberian tiger Panthera tigris altaica Korean magpie ( national bird ) Pica ( pica ) sericea Sri Lanka Sri Lankan junglefowl ( national bird ) Gallus
lafayettii Tanzania Giraffe Giraffa camelopardalis Thailand Elephant
</item 4>

<item 5>
List of national animals
Danphe ( national bird ) Lophophorus impejanus Netherlands Lion ( heraldic symbol ) Panthera leo Black - tailed godwit ( national bird ) Limosa limosa New
Zealand Kiwi Apteryx sp . North Korea Chollima Mythical Norway Lion ( royal national animal ) Panthera leo White - throated dipper ( national bird ) Cinclus
cinclus Fjord horse ( national horse ) Equus ferus caballus Oman Arabian oryx ( national animal ) Oryx leucoryx Pakistan Markhor ( National animal ) Capra
falconeri Indus river dolphin ( National aquatic mammal ) Platanista indicus minor minor Chukar ( National bird ) Alectoris chukar Indus crocodile ( National
reptile ) Crocodylus palustris Shaheen falcon ( Heritage bird ) Falco peregrinus peregrinator Snow leopard ( National predator ) Panthera uncia Panama Harpy
eagle Harpia harpyja Papua New Guinea Dugong ( national marine mammal ) Dugong dugon Peru Vicuña ( national animal ) Vicugna vicugna Philippines Carabao (
national animal ) Bubalus bubalis Philippine
</item 5>

<item 6>
List of national animals
versicolor Carp ( national fish ) Cyprinus carpio Latvia White wagtail ( national bird ) Motacilla alba Two - spotted ladybird ( national insect ) Adalia
bipunctata Lithuania White stork Ciconia ciconia Macedonia Lion ( in Macedonian heraldry ) Panthera leo Madagascar Zebu ( in Malagasy heraldry ) Bos primigenius
indicus Ring - tailed lemur ( national animal ) Lemur catta Malaysia Malayan tiger ( national animal ) Panthera tigris jacksoni Malta Pharaoh Hound ( national
animal ) Canis lupus familiaris ( 1 ) Moldova Aurochs ( national animal ) Bos primigenius Mauritius Dodo Raphus cucullatus Mexico Golden eagle ( national animal
) Aquila chrysaetos Xoloitzcuintli ( national dog ) Canis lupus familiaris Jaguar ( national mammal ) Panthera onca Grasshopper ( national arthropod )
Sphenarium purpurascens Vaquita ( national marine mammal ) Phocoena sinus Nepal Cow ( national animal ) Bos taurus indicus Danphe ( national bird ) Lophophorus
impejanus Netherlands Lion ( heraldic symbol
</item 6>

<item 7>
List of national animals
Jump up ^ Jon Eldridge . `` Primeval Poland : where the bison roam '' . theguardian.com . Retrieved 2016 - 10 - 10 . Jump up ^ `` Poland '' . allaboutbison.com
. Retrieved 2016 - 10 - 10 . Jump up ^ Tamra Orr ( 30 June 2008 ) . Qatar . Marshall Cavendish . p. 13 . ISBN 978 - 0 - 7614 - 2566 - 3 . Retrieved 30 July 2011
. Jump up ^ `` Animalul national - Rasul - Sapte simboluri nationale ale Romaniei mai putin cunoscute '' . 9AM . Jump up ^ `` Birobidjan , Russia Beaverton 's
sister city since 1990 '' . The City of Beaverton . Retrieved 2017 - 01 - 23 . Jump up ^ `` Rwanda State Symbols , Song , Flag , and More '' . worldatlas.com .
Retrieved 2013 - 03 - 21 .
</item 7>

<item 8>
List of national animals
List of National Animals - wikipedia List of National Animals Jump to : navigation , search This article is incomplete . Please help to improve it , or discuss
the issue on the talk page . ( June 2017 ) This is a list of national animals . National animals Country Name of animal Scientific name Pictures Ref . Algeria
Fennec fox Vulpes zerda Angola Red - crested turaco ( national bird ) Tauraco erythrolophus Anguilla Zenaida dove Zenaida aurita Antigua and Barbuda Fallow deer
( national animal ) Dama dama Frigate ( national bird ) Fregata magnificens Hawksbill turtle ( national sea creature ) Eretmochelys imbricata Argentina Rufous
hornero Furnarius rufus Australia Red kangaroo ( national animal ) Macropus rufus Emu ( national bird ) Dromaius novaehollandiae Austria Black eagle Ictinaetus
malaiensis Azerbaijan Karabakh horse Equus ferus caballus Bangladesh Royal Bengal tiger ( national animal ) Panthera tigris tigris Magpie robin ( national bird
) Copsychus saularis Ilish ( national fish )
</item 8>

<item 9>
List of national animals
bird ) Gallus lafayettii Tanzania Giraffe Giraffa camelopardalis Thailand Elephant Elephas maximus United Arab Emirates Arabian oryx ( national animal ) Oryx
leucoryx Peregrine falcon ( national bird ) Falco peregrinus United Kingdom Lion ( United Kingdom , England , Scotland , Wales ) Panthera leo Bulldog ( Britain
as `` British Bulldog '' ) Canis lupus familiaris Red kite ( Wales ) Milvus milvus Unicorn ( Scotland ) Mythical Y Ddraig Goch ( Welsh Dragon Wales ) Mythical
United States Bald eagle ( national bird ) Haliaeetus leucocephalus American bison ( national mammal ) Bison bison See also Animals portal Lists portal State
animal List of national birds Animals as heraldic charges Floral emblem National personification References Jump up ^ `` The Fennec Fox : Algeria 's National
Animal '' . algeria.com . Retrieved 2013 - 03 - 21 . Jump up ^ `` THE NATIONAL BIRD OF ANGOLA '' . BIRDS ANGOLA . Retrieved 12 April 2016 .
</item 9>

<item 10>
List of national animals
Archived from the original on 2013 - 07 - 30 . Retrieved 2013 - 08 - 29 . Jump up ^ Minahan , James ( 2009 ) . The Complete Guide to National Symbols and
Emblems : Volume 2 . Greenwood Press . p. 908 . ISBN 0313345007 . ^ Jump up to : `` South Africa 's national symbols '' . SouthAfrica.info . Brand South Africa
. Archived from the original on 14 December 2011 . Retrieved 9 February 2012 . ^ Jump up to : `` South Korean Symbols '' . worldatlas . Retrieved 12 April 2016
. Jump up ^ `` Sri Lanka National Bird '' . Government of Sri Lanka Official Web Portal . Archived from the original on 14 March 2012 . Jump up ^ `` National
Symbols : National Animal '' . tanzania.go.tz . Tanzania Government Portal . Retrieved 14 January 2015 .
</item 10>
=== END ITEM id=9119f762f458cbb4 ===

=== ITEM id=5379196307951b9d ===
Gold sentences and facts:
Sentence 0: The trade winds are the prevailing pattern of easterly surface winds found in the tropics , within the lower portion of the Earth 's atmosphere , in the lower section of the troposphere near the Earth 's equator . The trade winds blow predominantly from the northeast in the Northern Hemisphere and from the southeast in the Southern Hemisphere , strengthening during the winter and when the Arctic oscillation is in its warm phase . Trade winds have been used by captains of sailing ships to cross the world 's oceans for centuries , and enabled European empire expansion into the Americas and trade routes to become established across the Atlantic and Pacific oceans .
  Fact 0.0: Trade winds are the prevailing pattern of easterly surface winds found in the tropics.
  Fact 0.1: Trade winds are found within the lower portion of the Earth's atmosphere.
  Fact 0.2: Trade winds are found in the lower section of the troposphere near the Earth's equator.
  Fact 0.3: Trade winds blow predominantly from the northeast in the Northern Hemisphere.
  Fact 0.4: Trade winds blow predominantly from the southeast in the Southern Hemisphere.
  Fact 0.5: Trade winds strengthen during the winter.
  Fact 0.6: Trade winds strengthen when the Arctic oscillation is in its warm phase.
  Fact 0.7: Trade winds have been used by captains of sailing ships to cross the world's oceans for centuries.
  Fact 0.8: Trade winds enabled European empire expansion into the Americas.
  Fact 0.9: Trade winds enabled trade routes to become established across the Atlantic and Pacific oceans.

Retrieved items:
<item 1>
Trade winds
The subsident ( sinking ) air is relatively dry because as it descends , the temperature increases , but the absolute humidity remains constant , which lowers
the relative humidity of the air mass . This warm , dry air is known as a superior air mass and normally resides above a maritime tropical ( warm and moist )
air mass . An increase of temperature with height is known as a temperature inversion . When it occurs within a trade wind regime , it is known as a trade wind
inversion . The surface air that flows from these subtropical high - pressure belts toward the Equator is deflected toward the west in both hemispheres by the
Coriolis effect . These winds blow predominantly from the northeast in the Northern Hemisphere and from the southeast in the Southern Hemisphere . Because winds
are named for the direction from which the wind is blowing , these winds are called the northeasterly trade winds in the Northern Hemisphere and the
southeasterly trade winds in the Southern Hemisphere . The trade winds of both hemispheres meet at the doldrums . As they blow across tropical regions , air
masses heat up over lower latitudes due to more direct sunlight .
</item 1>

<item 2>
Trade winds
Encyclopedia of world climatology . Springer . p. 128 . ISBN 978 - 1 - 4020 - 3264 - 6 . Retrieved 2009 - 11 - 09 . Jump up ^ Bob Rauber ( 2009 - 05 - 22 ) . ``
Research - The Rain in Cumulus over the Ocean Campaign '' . Retrieved 2009 - 11 - 08 . Jump up ^ James P. Terry ( 2007 ) . Tropical cyclones : climatology and
impacts in the South Pacific . Springer . p. 8 . ISBN 978 - 0 - 387 - 71542 - 1 . Retrieved 2009 - 11 - 08 . Jump up ^ G.E. Pieter & F. Augustinus . `` The
influence of the trade winds on the coastal development of the Guianas at various scale levels : a synthesis '' . Marine Geology. 208 ( 2 - 4 ) : 145 -- 151 .
Bibcode : 2004MGeol. 208 ... 145A .
</item 2>

<item 3>
Trade winds
`` trade winds '' . Glossary of Meteorology . American Meteorological Society . Archived from the original on 2008 - 12 - 11 . Retrieved 2008 - 09 - 08 . Jump
up ^ Ralph Stockman Tarr and Frank Morton McMurry ( 1909 ). Advanced geography . W.W. Shannon , State Printing , pp. 246 . Retrieved on 2009 - 04 - 15 . Jump up
^ JetStream ( 2008 ) . `` How to read weather maps '' . National Weather Service . Retrieved 2007 - 05 - 16 . Jump up ^ Glossary of Meteorology ( June 2000 ) .
`` Tropical air '' . American Meteorological Society . Retrieved 2009 - 10 - 28 . Jump up ^ Glossary of Meteorology ( June 2000 ) . `` Trade air '' . American
Meteorological Society . Retrieved 2009 - 10 - 28 . Jump up ^ John E. Oliver ( 2005 ) . Encyclopedia of world climatology . Springer .
</item 3>

<item 4>
Trade winds
ISBN 978 - 1 - 4437 - 2300 - 8 . Retrieved 2009 - 11 - 09 . Jump up ^ Glossary of Meteorology ( June 2000 ) . `` Doldrums '' . American Meteorological Society .
Retrieved 2009 - 11 - 09 . Jump up ^ Glossary of Meteorology ( June 2000 ) . `` Intertropical Convergence Zone '' . American Meteorological Society . Retrieved
2009 - 11 - 09 . Jump up ^ Glossary of Meteorology ( June 2000 ) . `` Monsoon Trough '' . American Meteorological Society . Archived from the original on 2009 -
06 - 17 . Retrieved 2009 - 11 - 09 . Jump up ^ Glossary of Meteorology ( June 2000 ) . `` Superior air '' . American Meteorological Society . Retrieved 2009 -
10 - 28 . Jump up ^ Glossary of Meteorology ( 2009 ) . `` trade winds '' . Glossary of Meteorology .
</item 4>

<item 5>
Trade winds
Those that develop over land ( continental ) are drier and hotter than those that develop over oceans ( maritime ) , and travel northward on the western
periphery of the subtropical ridge . Maritime tropical air masses are sometimes referred to as trade air masses . The one region of the Earth which has an
absence of trade winds is the north Indian ocean . Weather effects ( edit ) Clouds which form above regions within trade wind regimes are typically composed of
cumulus which extend no more than 4 kilometres ( 13,000 ft ) in height , and are capped from being taller by the trade wind inversion . Trade winds originate
more from the direction of the poles ( northeast in the Northern Hemisphere , southeast in the Southern Hemisphere ) during the cold season , and are stronger
in the winter than the summer . As an example , the windy season in the Guianas , which lie at low latitudes in South America , occurs between January and April
. When the phase of the Arctic oscillation ( AO ) is warm , trade winds are stronger within the tropics . The cold phase of the AO leads to weaker trade winds .
</item 5>

<item 6>
Trade winds
Trade winds - wikipedia Trade winds Jump to : navigation , search This article is about the weather phenomenon . For other uses , see Tradewind . The westerlies
( blue arrows ) and trade winds ( yellow and brown arrows ) The trade winds are the prevailing pattern of easterly surface winds found in the tropics , within
the lower portion of the Earth 's atmosphere , in the lower section of the troposphere near the Earth 's equator . The trade winds blow predominantly from the
northeast in the Northern Hemisphere and from the southeast in the Southern Hemisphere , strengthening during the winter and when the Arctic oscillation is in
its warm phase . Trade winds have been used by captains of sailing ships to cross the world 's oceans for centuries , and enabled European empire expansion into
the Americas and trade routes to become established across the Atlantic and Pacific oceans . In meteorology , the trade winds act as the steering flow for
tropical storms that form over the Atlantic , Pacific , and southern Indian Oceans and make landfall in North America , Southeast Asia , and Madagascar and
eastern Africa , respectively . Trade winds also transport African dust westward across the Atlantic Ocean into the Caribbean Sea , as well as portions of
southeastern North America .
</item 6>

<item 7>
Trade winds
Between 1847 and 1849 , Matthew Fontaine Maury collected enough information to create wind and current charts for the world 's oceans . Cause ( edit ) 3D map
showing Hadley cells in relationship to trade winds on the surface . See also : Air mass , Hadley cell , Humidity , Intertropical Convergence Zone , Monsoon ,
Monsoon trough , Near - equatorial trough , and Prevailing winds As part of the Hadley cell circulation , surface air flows toward the equator while the flow
aloft is towards the poles . A low - pressure area of calm , light variable winds near the equator is known as the doldrums , near - equatorial trough ,
intertropical front , or the Intertropical Convergence Zone . When located within a monsoon region , this zone of low pressure and wind convergence is also
known as the monsoon trough . Around 30 ° in both hemispheres , air begins to descend toward the surface in subtropical high - pressure belts known as
subtropical ridges .
</item 7>

<item 8>
Trade winds
See also ( edit ) Intertropical Convergence Zone Winds in the Age of Sail Westerly wind burst Volta do mar References ( edit ) Jump up ^ Carol G. Braham ; Enid
Pearsons ; Deborah M. Posner ; Georgia S. Maas & Richard Goodman ( 2001 ) . Random House Webster 's College Dictionary ( second ed . ) . Random House . p. 1385
. ISBN 0 - 375 - 42560 - 8 . Jump up ^ Hermann R. Muelder ( 2007 ) . Years of This Land - A Geographical History of the United States . Read Books . p. 38 .
ISBN 978 - 1 - 4067 - 7740 - 6 . Retrieved 2009 - 11 - 09 . ^ Jump up to : Derek Hayes ( 2001 ) . Historical atlas of the North Pacific Ocean : maps of
discovery and scientific exploration , 1500 - 2000 . Douglas & McIntyre . p. 18 . ISBN 978 - 1 - 55054 - 865 - 5 .
</item 8>

<item 9>
Trade winds
They also learned that to reach South Africa , they needed to go far out in the ocean , head for Brazil , and around 30 ° S go east again . Following the
African coast southbound means upwind in the Southern hemisphere . In the Pacific ocean , the full wind circulation , which included both the trade wind
easterlies and higher - latitude Westerlies , was unknown to Europeans until Andres de Urdaneta 's voyage in 1565 . The captain of a sailing ship seeks a course
along which the winds can be expected to blow in the direction of travel . During the Age of Sail , the pattern of prevailing winds made various points of the
globe easy or difficult to access , and therefore had a direct effect on European empire - building and thus on modern political geography . For example ,
Manila galleons could not sail into the wind at all . By the 18th century the importance of the trade winds to England 's merchant fleet for crossing the
Atlantic Ocean had led both the general public and etymologists to identify the name with a later meaning of ' trade ' , `` ( foreign ) commerce '' .
</item 9>

<item 10>
Trade winds
Shallow cumulus clouds are seen within trade wind regimes , and are capped from becoming taller by a trade wind inversion , which is caused by descending air
aloft from within the subtropical ridge . The weaker the trade winds become , the more rainfall can be expected in the neighboring landmasses . Contents ( hide
) 1 History 2 Cause 3 Weather effects 4 See also 5 References History ( edit ) A Spanish galleon See also : Age of Discovery , Volta do Mar , and Age of sail
The term trade winds originally derives from the early fourteenth century late Middle English word ' trade , ' meaning `` path '' or `` track . '' The
Portuguese recognized the importance of the trade winds ( then the Volta do mar , meaning in Portuguese `` turn of the sea '' but also `` return from the sea ''
) in navigation in both the north and south Atlantic ocean as early as the 15th century . From West Africa , the Portuguese had to sail away from continental
Portugal , that is , to west and northwest . They could then turn northeast , to the area around the Azores islands , and finally east to mainland Europe .
</item 10>
=== END ITEM id=5379196307951b9d ===

=== ITEM id=31f44c28761e3456 ===
Gold sentences and facts:
Sentence 0: No . Official English title Original Japanese title Original air date English air date 151 `` Sabertooth '' `` Seibātūsu '' ( 剣 咬 の 虎 ( セイバー トゥース ) ) October 6 , 2012 November 17 , 2015 The Legion Platoon is tasked by Zentopia 's archbishop to retrieve the missing pieces of the Infinity Clock and seal them again . Before embarking on their mission , they visit Fairy Tail and make amends with the guild for their previous hostility against them . It is later discovered that Fairy Tail has lost their title as the strongest guild in Fiore to the Saber Tooth guild over the past seven years . Two of the strongest wizards in the guild include a pair of Dragon Slayers named Sting and Rogue , who are obsessed with fighting Natsu and Gajeel , respectively . Meanwhile , Makarov brings Gildarts to their abandoned guildhall and shows him Lumen Histoire , a light that Makarov says is Fairy Tail 's greatest secret . Makarov then tells Gildarts that he has been chosen to become the next master of Fairy Tail . 152 `` And So We Aim for the Top '' `` Soshite Ore - tachi wa Chōjō o Mezasu '' ( そして オレ たち は 顶 上 を 目指す ) October 13 , 2012 November 24 , 2015 Natsu and his friends discover that a power gap has developed between them and the rest of the guild members who were not on Tenrou Island . The group meets with Poluchka for medicine to strengthen their magic , but are unsuccessful . Wendy recognizes Poluchka 's voice and scent as those of Grandine , the dragon who raised her . Poluchka reveals herself as the Edolas counterpart of Grandine , and gives Wendy written instructions for two advanced spells that Grandine failed to teach her . Meanwhile , Makarov announces Gildarts ' succession as master to the rest of the guild , only to find he has gone on a journey , leaving a letter of resignation behind . In the letter , Gildarts reinstates Laxus into the guild , returns mastership to Makarov , and tells him to make Fairy Tail the strongest guild in Fiore again . Romeo suggests they participate in the Grand Magic Games , an annual festival designed to determine the guild rankings in Fiore . 153 `` Song of the Stars '' `` Hoshiboshi no Uta '' ( 星々 の 歌 ) October 20 , 2012 December 1 , 2015 With the Grand Magic Games occurring in three months time , Natsu and his friends train at a beach resort . On their second day of training , Virgo appears and tells the wizards that the Celestial Spirit World is in danger of destruction . When the wizards arrive , they instead find a surprise party held by the Celestial Spirit King celebrating their return from Sirius Island after seven years . They spend the day in the spirit world partying with the spirits . Before they leave , however , Virgo informs them that one day in the spirit world equals three months in the human world , and they find the Grand Magic Games are upon them . 154 `` For All the Time We Missed Each Other '' `` Surechigatta Jikan no Bun dake '' ( すれ 违 っ た 时间 の 分 だけ ) October 27 , 2012 December 8 , 2015 While mulling over their lost training time , the Fairy Tail wizards receive a message via carrier pigeon requesting a private meeting in the forest nearby . There they meet a runaway Jellal , Ultear , and Merudy , who have formed their own guild , Crime Sorcière , to defeat Zeref and any dark guilds that try to use his power . The three ask Fairy Tail for their help in determining the source of a magical anomaly similar to Zeref 's magic that they have sensed around the Grand Magic Games arena , Domus Flau . In exchange , Ultear uses her magic to invoke their Second Origin in order to double their magic power . As the others undergo the painful process of awakening their Second Origin , Erza talks with Jellal over his regained memories . The two nearly kiss , but Jellal decides that he and Erza can not be together and lies to her that he has a fiancée . Erza quickly picks up on this and respects his decision , seeing Crime Sorcière off as they leave . 155 `` Crocus , the Flower - Blooming Capital '' `` Hanasaku Miyako : Kurokkasu '' ( 花 笑 く 都 ・ クロッカス ) November 3 , 2012 December 15 , 2015 Natsu , Lucy , Gray , Erza , and Wendy are chosen to represent their guild in the Grand Magic Games , which are to be held in Fiore 's capital , Crocus . As they tour the city , Natsu and Happy encounter the twin dragon slayers , Sting and Rogue , who mock Natsu for not being able to kill Acnologia , and boast of becoming true dragon slayers by killing the dragons who raised them . At midnight , Fairy Tail and the other participating guilds are put through a preliminary event -- a race through a massive floating labyrinth -- designed to narrow the number of participating guilds down to eight teams . 156 `` Sky Labyrinth '' `` Sukai Rabirinsu '' ( 空中 迷宮 ( スカイ ラビリンス ) ) November 10 , 2012 December 22 , 2015 Natsu 's team notices that Wendy has gone missing , so Elfman replaces her to keep them from being eliminated . As they make their way through the enchanted labyrinth , the five encounter various other guilds and steal the maps made by the other guilds to better navigate the maze . They manage to reach the end of the maze and pass the elimination round , confident they have arrived first , only to discover they have barely taken last place . Meanwhile , Happy , Lisanna , and the rest of the guild search for Wendy and Carla . They eventually find the two in the garden of the king 's palace , Mercurius , with their magic power depleted . 157 `` New Guild '' `` Shinki Girudo '' ( 新 规 ギルド ) November 17 , 2012 December 29 , 2015 Wendy 's body is weakened as a result of losing her magic power . As Poluchka helps her recover , she formally requests Elfman to take her place in the Grand Magic Games . Fairy Tail steps into the arena with a negative audience reception , but they are bolstered by their guild mates and the spirit of Mavis , whom only the Fairy Tail guild members can see . The next four participating teams are revealed to be Quatro Cerberus , Mermaid Heel , Blue Pegasus , and Lamia Scale . Following them is Raven Tail , a former dark guild led by Makarov 's son Ivan , who confess to attacking Wendy and Carla . To everyone 's shock , the next team is revealed to be a second team from Fairy Tail composed of Gajeel , Juvia , Mirajane , Laxus , and Jellal ( disguised as Mystogan to help investigate the arena ) . The final guild to be announced is Saber Tooth , much to the joy of the audience . With all eight teams announced , they each send out one member to participate in the first contest of the games , with Gray and Juvia representing their teams . 158 `` Night of Shooting Stars '' `` Hoshi Furu Yoru ni '' ( 星 降 ル 夜 ニ ) November 24 , 2012 January 5 , 2016 The games ' first contest , `` Hidden '' , requires the eight participants to score points by finding and attacking each other in a virtual city filled with copies of themselves . Raven Tail member Nalpudding repeatedly targets Gray , who scores last place in the contest while Saber Tooth 's Rufus scores first by using his Memory - Make magic to attack all the other participants at once . Humiliated , Gray swears revenge against Raven Tail and Saber Tooth . Afterwards , Lucy is chosen to fight Raven Tail member Flare in the tournament 's first battle round . 159 `` Lucy vs. Flare '' `` Rūshii vs. Furea '' ( ルーシィ vs. フレア ) December 1 , 2012 January 12 , 2016 Lucy displays tremendous skill in her battle against Flare by being able to summon two spirits simultaneously . As Lucy gains the upper hand , Flare cheats by using her magic hair to hold Alzack and Bisca 's daughter Asuka as a hostage , blackmailing Lucy into not fighting back to keep her from harming Asuka . As Lucy tearfully speaks of Asuka 's danger , Natsu is able to hear it and rushes to Asuka to incinerate Flare 's hair , allowing Lucy to counterattack with her Urano Metoria spell in combination with Gemini . However , before the spell fully activates , Flare 's teammate Obra nullifies Lucy 's magic , causing her to collapse and lose the match . 160 `` Portent '' `` Kyōzui '' ( 凶 瑞 ) December 8 , 2012 January 19 , 2016 Blue Pegasus ' Ren defeats Mermaid Heel 's Araña during the games ' second match , while Saber Tooth 's Orga effortlessly defeats Quatro Cerberus ' War Cry in the third match . For the day 's fourth and final match , Jellal is pitted against Lamia Scale 's Jura , intending to avenge the humiliated Fairy Tail members . He mimics Mystogan 's magic to keep up his disguise , but he finds it ineffective to use against Jura and is forced to use his own magic instead . Before he can cast a spell that will expose his true identity , Ultear uses Merudy 's sensory to force Jellal into submission . Meanwhile , the hospitalized Carla awakens after having a troubling premonition of the future . 161 `` Chariots '' `` Chariotto '' ( 战 车 ( チャリオット ) ) December 15 , 2012 January 26 , 2016 Fairy Tail is undaunted by their poor performance on the first day . As the guild celebrates , they are visited and taunted by Bacchus , Erza 's rival from Quatro Cerberus , who has come to replace War Cry . Carla relays her premonition to Poluchka , saying that she saw Mercurius collapsing around what appeared to be a tearfully singing Lucy . The second day of the tournament begins with a footrace atop a train of moving chariots , causing the participants Natsu , Gajeel , and Sting to suffer from motion sickness as a side effect of their Dragon Slayer magic . As Natsu and Gajeel force themselves forward , Sting inquires about their determination to which Natsu responds that it is as a result of the strength of Fairy Tail 's bonds . Sting eventually throws the race while Natsu and Gajeel manage to finish as sixth and seventh placers , respectively , earning Fairy Tail the audience 's respect . 162 `` Elfman vs. Bacchus '' `` Erufuman vs. Bakkasu '' ( エルフマン vs. バッカス ) December 22 , 2012 September 1 , 2015 Due to a miscomunnication between the king and his knight Arcadios , Elfman is chosen as Bacchus ' opponent instead of Erza . Before the match commences , the two opponents make a wager : if Bacchus wins , he will court Elfman 's sisters , and if Elfman wins , Bacchus ' team name will be changed to `` Quatro Puppy '' for the rest of the games . Bacchus ruthlessly attacks Elfman with his bare palms , who manages to damage Bacchus with his defensive Lizardman transformation . Elfman manages to outlast Bacchus as he exhausts himself , winning the match and forcing Bacchus ' team to be renamed . Meanwhile , Natsu finds Wendy , Carla , and Poluchka being kidnapped by a group of masked men . Upon being caught , the men claim they were hired to kidnap Lucy by Raven Tail . They are soon revealed to have actually been sent by Arcadios , who intends to sow discord between the guilds as he plots to use Lucy for his `` Eclipse Project '' . 163 `` Mirajane vs. Jenny '' `` Mirajēn vs. Jenī '' ( ミラ ジェーン vs. ジェニー ) January 5 , 2013 September 1 , 2015 The second day of the Grand Magic Games continues , with Mirajane of Fairy Tail Team B fighting against Jenny Realight of Team Blue Pegasus . Since the two competitors are well - known for being magazine models , their battle consists almost entirely of a fashion showdown , much to the pleasure of the audience . Eventually as the showdown becomes redundant , Jenny proposes a wager : the loser must pose nude for the Sorcerer Magazine , which Mirajane accepts . Jenny then transforms into her battle armor while Mirajane shifts into her Satan Soul `` Sitri '' form and easily defeats Jenny . The final match of the day is announced , with Kagura Mikazuchi of Mermaid Heel pitted against Yukino Aguria of Saber Tooth . Yukino proposes to Kagura that they wager their lives on their battle . Meanwhile , Arcadios is confronted by Darton , the kingdom 's defense minister , about his `` Eclipse Project '' . 164 `` Kagura vs. Yukino '' `` Kagura vs. Yukino '' ( カグラ vs. ユキノ ) January 12 , 2013 September 1 , 2015 Yukino is revealed to be a celestial spirit wizard when she summons Libra and Pisces to attack Kagura . Using her gravity - altering magic , Kagura defeats the two spirits , thus forcing Yukino to use a Gatekey that summons Ophiuchus , which is regarded as the strongest among the Celestial Spirits . However , Kagura slays Ophiuchus with her sword still sheathed and easily defeats Yukino . Later , Jellal discusses the fact that he has not sensed the evil power similar to Zeref 's even though the second day of the Grand Magic Games has ended . Erza reunites with Milliana , who is revealed to be a member of Mermaid Heel . Meanwhile , Saber Tooth 's master Jiemma is dissatisfied with his guild 's performance . Though Sting is given another chance , Jiemma humiliates Yukino for her failure . 165 `` Hatred at Nightfall '' `` Urami wa Yoru no Tobari ni Tsutsumarete '' ( 怨み は 夜 の 帐 に 包 まれ て ) January 19 , 2013 September 1 , 2015 After Yukino is expelled from Saber Tooth , she offers her two golden Gatekeys to Lucy to complete the latter 's collection of the Zodiac Gatekeys , stating that when the twelve Zodiac Gatekeys are all gathered , a gateway that changes the world will open ; however , Lucy turns down the offer . Meanwhile , Milliana tells Erza that Kagura 's sword , Archenemy , is intended to kill Jellal , and that she and Kagura share the same resentment towards Jellal . Natsu and Happy run after Yukino in an attempt to apologize ; however , Natsu realizes that Yukino had been expelled , and infiltrates the Crocus Garden to fight Jiemma . 166 `` Pandemonium '' `` Pandemoniumu '' ( 伏魔殿 ( パンデモニウム ) ) January 26 , 2013 September 1 , 2015 After Natsu punches out Dobengal , the guild member summoned to fight him , he and Jiemma engage in a duel . However , the guild master 's daughter Minerva appears , ends their duel and , taking Happy hostage , convinces Natsu to settle their dispute in the Grand Magic Games . As Natsu leaves , he states that Saber Tooth will never surpass Fairy Tail because they care for their friends , words that are echoed by Rogue , while Sting is left in awe at Natsu 's strength . The following day as the Grand Magic Games continue , an event called `` Pandemonium '' begins . It involves the participants entering a magical citadel filled with 100 monsters of varying strengths and individually doing battle with any number of monsters they choose . Representing Fairy Tail Team A , Erza requests the 100 monsters as her opponents . 167 `` 100 Against 1 '' `` Hyaku tai Ichi '' ( 100 対 1 ) February 2 , 2013 September 1 , 2015 Erza kills the 100 monsters using various types of armor to match their strength and weaknesses . As a result , the crowd praises Fairy Tail , and the guild is awarded with 10 points . Moreover , the game officials decided to determine the ranks of the remaining participants by means of a device called `` Magic Power Finder '' ( sometimes abbreviated as MPF ) . Using this tool , participants must attack it with their magic power , which will , in turn , calculate and display their equivalent scores depending on the extent of power used . In the end , Cana of Fairy Tail Team B dominates the event by using the Fairy Glitter spell lent to her by Mavis , much to Makarov 's surprise . With Cana 's victory , Fairy Tail Team B receives 8 points , and the audience began to burst once more into a resounding swarm of cheers and claps due to the finish of the two Fairy Tail teams . Meanwhile , the Raven Tail team somehow managed to get an insider information regarding the battle matchups for the day , so the masked leader Alexei tells his subordinates Flare , Obra , Nalpudding , and Kurohebi that it 's about time to proceed with their true mission . 168 `` Laxus vs. Alexei '' `` Rakusasu vs. Arekusei '' ( ラクサス vs. アレクセイ ) February 9 , 2013 September 1 , 2015 In the battle portion of the Grand Magic Games ' third day , Milliana of Mermaid Heel beats Semmes of Quatro Puppy while Saber Tooth 's Rufus swiftly defeats Blue Pegasus ' Eve . For the third match , Laxus of Fairy Tail Team B is chosen to battle against Alexei of Team Raven Tail . Since the opponent is from a suspicious guild , the Fairy Tail members take extra precautions by watching the actions of Ivan Dreyar as well as the other four Raven Tail members . However , as the match commences , the spectators are shocked to see Laxus being beaten by Alexei , although no cautious movements can be seen from the Raven Tail team . It turns out that the two dueling people are just illusions created by Alexei in order for him to have a talk with Laxus . Also , the other Raven Tail members being monitored by Fairy Tail are just lifelike illusions , for in reality they are with Alexei in the battlefield . Alexei reveals himself to be Ivan Dreyar . Unable to interrogate Laxus of the Lumen Histoire 's location , Ivan and the Raven Tail members are forced to fight him . Before their battle begins , Ivan declares that Raven Tail is an anti-Fairy Tail guild . 169 `` Wendy vs. Shelia '' `` Wendi vs. Sheria '' ( ウェンディ vs. シェリア ) February 16 , 2013 September 1 , 2015 After Laxus defeats Ivan and the four Raven Tail elite members , Ivan 's illusions disappear to the shock of everyone in the stadium . The game officials declare Laxus the victor , and Ivan tells him that Lumen Histoire is Fairy Tail 's ' darkness ' . Due to the infraction of the rules , the Raven Tail team is disqualified for the remainder of the Games , and its members are captured by the soldiers . Later , Wendy of Fairy Tail Team A and Sherria Blendy of Team Lamia Scale face off to compete for the final match of the day . As the two exchange attacks , gusts of wind surround the arena since Sherria is revealed to be a Sky God Slayer . Wendy employs an advanced Sky Dragon Slayer attack much to Poluchka 's astonishment , causing Sherria to fall on the ground . Wendy is about to be announced as the winner of the bout when Sherria suddenly stands up with her wounds fully healed . Meanwhile , the Crime Sorcière guild senses the nefarious aura , and Jellal proceeds to the Domus Flau audience , oblivious that he bumped into Doranbolt , who , in turn , wonders what is Mystogan doing in Earthland . 170 `` Small Fists '' `` Chiisana Kobushi '' ( 小さな 拳 ) February 23 , 2013 September 1 , 2015 With Cheria being fully healed and Wendy being almost depleted of magic power , their battle continues , with the former casting a high - level spell towards the latter . However , it seems that Wendy enhances Cheria 's physical strength , causing the attack to gain much force and thus missing Wendy . The two competitors then engage in a hand - to - hand combat , but due to the time limit and neither side is giving up , their battle resulted in a draw , giving both teams with 5 points . Meanwhile , Jellal manages to track the sinister aura among the crowd in the form of a mysterious , hooded person . Jellal is about to confront the aura source , but he is abruptly cornered by Doranbolt , Lahar , and other Rune Knights . Doranbalt then interrogates Jellal , and as his cover was blown , the Rune Knights , as well as Kagura and Milliana glancing nearby , are startled to see the true identity of Mystogan . Yajima convinces the Rune Knights that Mystogan is Jellal 's Edolas counterpart , prompting Lahar to release the fugitive . Meanwhile , Laxus and Makarov have a small conversation regarding Lumen Histoire when Mavis appears and explains that Lumen Histoire is the ' light ' of Fairy Tail . 171 `` Naval Battle '' `` Nabaru Batoru '' ( 海战 ( ナバルバトル ) ) March 2 , 2013 September 1 , 2015 As the jubilant Fairy Tail members celebrate their victories , the other people , such as Sting , Kagura , and Milliana , are contemplating on their own concerns , while Yukino is enlisted as a sergeant under Arcadios ' command . Meanwhile , the fourth day of the competitions comes to a start with an event , `` Naval Battle '' , which involves the participants swimming in a giant water sphere and knocking the other competitors out . After the other participants fall out of the aquatic sphere , a five - minute battle between Lucy and Minerva ensues . However , Minerva quickly attacks Lucy , and she manages to steal the keys of the Celestial Spirit mage . Regardless of this , Lucy refuses to surrender . After hearing Lucy 's intentions , Minerva stops from attacking up to the extent that the five - minute mark was reached . However , Minerva resumes attacking Lucy until the game officials decide to halt the match . The Fairy Tail members snap , and Minerva dangles Lucy out of the sphere while the others rush to her aid . 172 `` A Parfum for You '' `` Kimi ni Sasageru Parufamu '' ( 君 に 捧げる 香り ( パルファム ) ) March 9 , 2013 September 1 , 2015 While Lucy is hospitalized and taken to the infirmary , the Fairy Tail members warn Saber Tooth that they have angered the worst possible guild to do so . Later , Makarov announces that the two Fairy Tail teams must merge as a result of the disqualification of Raven Tail and to compensate for the odd number of teams for the battle round . He also reveals that the new team will have the former Team A 's 35 points to start with . The new Fairy Tail team , consisting of Natsu , Gray , Erza , Gajeel , and Laxus , are well - received as they enter the field . The second part of the tournament begins with a tag battle between Blue Pegasus ' Ichiya and `` Rabbit '' and Quatro Puppy 's Rocker and Bacchus . To everyone 's shock , it is revealed that `` Rabbit '' is actually Nichiya in disguise , and after the Exceed is punched by Bacchus , Ichiya bulks up to defeat the Quatro Puppy opponents , awarding Blue Pegasus with 10 points . 173 `` Battle of Dragon Slayers ! '' `` Batoru obu Doragon Sureiyā '' ( バトル ・ オブ ・ ドラゴン スレイヤー ) March 16 , 2013 September 1 , 2015 Mermaid Heel 's Kagura and Milliana and Lamia Scale 's Lyon and Yuka face off for the second tag battle of the games . The two teams struggle to earn a victory , only to have their battle ending in a draw due to the time limit . For the third and final match of the day , the spectators look on in silence as Natsu and Gajeel of Team Fairy Tail and Sting and Rogue of Team Saber Tooth enter the field . As the battle commences , Sting and Rogue were about to take the initiative when they were suddenly attacked by Natsu and Gajeel . The four mages then exchange attacks ; however , the audience watches in shock as the Fairy Tail Dragon Slayers overpower the Saber Tooth Dragon Slayers . Unable to keep themselves on par with their adversaries , Sting activates ' White Drive ' while Rogue powers up using ' Shadow Drive ' . 174 `` Four Dragons '' `` Yonin no Doragon '' ( 四 人 の 竜 ( ドラゴン ) ) March 23 , 2013 September 1 , 2015 The tag battle continues , with Sting and Rogue beginning to attack Natsu and Gajeel with their powers being increased . After receiving heavy blows from the Saber Tooth Dragon Slayers , Natsu and Gajeel manages to hold them off , thus forcing Sting and Rogue to activate `` Dragon Force '' on their own will , to the shock of the Fairy Tail audience . Sting decides to finish the Fairy Tail Dragon Slayers alone , for he held a promise to his Exceed friend Lector many years ago . As the three mages fight , the tide turns in favor of Sting as he unleashes a powerful attack on Natsu and Gajeel , resulting in the collapse of the arena floor , and after a hand - to - hand combat underground , Sting was seemingly the victor of the match . However , the Fairy Tail duo stands up , comments on Sting 's strength , and argues about his position while attacking , thereby causing Natsu to drag Gajeel on a mining cart and send it away . While Gajeel is left incapacitated due to motion sickness , Natsu challenges the Twin Dragons of Saber Tooth alone , in retribution to what Sting did earlier on them . 175 `` Natsu vs. the Twin Dragons '' `` Natsu vs. Sōryū '' ( ナツ vs. 双竜 ) March 30 , 2013 September 1 , 2015 The tag battle of the Dragon Slayers comes to a close , with Natsu overpowering the combined attacks of Sting and Rogue . The Saber Tooth members watch in dismay , while Jiemma seethes in anger over the defeat of the Twin Dragons , frightening the nearby spectators away . The rest of Fairy Tail celebrates in joy as its team receives 10 points , finally surpassing Team Saber Tooth by a single point . Meanwhile , Gajeel , back to his senses after being dragged by Natsu into a cart earlier , stumbles upon a cave full of dragon skeletons . During that night , Jellal manages to corner the source of the dark aura , standing in disbelief at the identity of the mysterious , hooded person .
  Fact 0.0: Episode 151, 'Sabertooth', has the Japanese title 'Seibatusu' and originally aired on October 6, 2012.
  Fact 0.1: Episode 151 aired in English on November 17, 2015.
  Fact 0.2: The Legion Platoon is tasked by Zentopia's archbishop to retrieve the missing pieces of the Infinity Clock and seal them again.
  Fact 0.3: Before embarking on their mission, the Legion Platoon visits Fairy Tail.
  Fact 0.4: The Legion Platoon makes amends with Fairy Tail for their previous hostility against them.
  Fact 0.5: It is later discovered that Fairy Tail has lost its title as the strongest guild in Fiore to Saber Tooth over the past seven years.
  Fact 0.6: Sting and Rogue are two of the strongest wizards in Fairy Tail.
  Fact 0.7: Sting and Rogue are a pair of Dragon Slayers.
  Fact 0.8: Sting is obsessed with fighting Natsu.
  Fact 0.9: Rogue is obsessed with fighting Gajeel.
  Fact 0.10: Makarov brings Gildarts to their abandoned guildhall.
  Fact 0.11: Makarov shows Gildarts Lumen Histoire, a light that Makarov says is Fairy Tail's greatest secret.
  Fact 0.12: Makarov tells Gildarts that he has been chosen to become the next master of Fairy Tail.
  Fact 0.13: Episode 152, 'And So We Aim for the Top', has the Japanese title 'Soshite Ore-tachi wa Chojo o Mezasu' and originally aired on October 13, 2012.
  Fact 0.14: Episode 152 aired in English on November 24, 2015.
  Fact 0.15: Natsu and his friends discover that a power gap has developed between them and the rest of the guild members who were not on Tenrou Island.
  Fact 0.16: The group meets with Poluchka for medicine to strengthen their magic.
  Fact 0.17: The group is unsuccessful in strengthening their magic with the medicine.
  Fact 0.18: Wendy recognizes Poluchka's voice and scent as those of Grandine, the dragon who raised her.
  Fact 0.19: Poluchka reveals herself as the Edolas counterpart of Grandine.
  Fact 0.20: Poluchka gives Wendy written instructions for two advanced spells that Grandine failed to teach her.
  Fact 0.21: Makarov announces Gildarts' succession as master to the rest of the guild.
  Fact 0.22: Makarov finds that Gildarts has gone on a journey, leaving a letter of resignation behind.
  Fact 0.23: In the letter, Gildarts reinstates Laxus into the guild.
  Fact 0.24: In the letter, Gildarts returns mastership to Makarov.
  Fact 0.25: In the letter, Gildarts tells Makarov to make Fairy Tail the strongest guild in Fiore again.
  Fact 0.26: Romeo suggests they participate in the Grand Magic Games.
  Fact 0.27: The Grand Magic Games are an annual festival designed to determine the guild rankings in Fiore.
  Fact 0.28: Episode 153, 'Song of the Stars', has the Japanese title 'Hoshiboshi no Uta' and originally aired on October 20, 2012.
  Fact 0.29: Episode 153 aired in English on December 1, 2015.
  Fact 0.30: The Grand Magic Games are occurring in three months' time.
  Fact 0.31: Natsu and his friends train at a beach resort.
  Fact 0.32: On their second day of training, Virgo appears and tells the wizards that the Celestial Spirit World is in danger of destruction.
  Fact 0.33: When the wizards arrive, they find a surprise party held by the Celestial Spirit King.
  Fact 0.34: The party celebrates the wizards' return from Sirius Island after seven years.
  Fact 0.35: The wizards spend the day in the spirit world partying with the spirits.
  Fact 0.36: Virgo informs the wizards that one day in the spirit world equals three months in the human world.
  Fact 0.37: The wizards find that the Grand Magic Games are upon them.
  Fact 0.38: Episode 154, 'For All the Time We Missed Each Other', has the Japanese title 'Surechigatta Jikan no Bun dake' and originally aired on October 27, 2012.
  Fact 0.39: Episode 154 aired in English on December 8, 2015.
  Fact 0.40: While mulling over their lost training time, the Fairy Tail wizards receive a message via carrier pigeon.
  Fact 0.41: The message requests a private meeting in the forest nearby.
  Fact 0.42: At the meeting, the wizards meet a runaway Jellal, Ultear, and Merudy.
  Fact 0.43: Jellal, Ultear, and Merudy have formed their own guild, Crime Sorciere.
  Fact 0.44: Crime Sorciere aims to defeat Zeref and any dark guilds that try to use his power.
  Fact 0.45: Jellal, Ultear, and Merudy ask Fairy Tail for help in determining the source of a magical anomaly similar to Zeref's magic.
  Fact 0.46: The magical anomaly has been sensed around the Grand Magic Games arena, Domus Flau.
  Fact 0.47: In exchange, Ultear uses her magic to invoke Fairy Tail's Second Origin in order to double their magic power.
  Fact 0.48: Erza talks with Jellal over his regained memories.
  Fact 0.49: Erza and Jellal nearly kiss.
  Fact 0.50: Jellal decides that he and Erza can not be together.
  Fact 0.51: Jellal lies to Erza that he has a fiancee.
  Fact 0.52: Erza quickly picks up on the lie and respects his decision.
  Fact 0.53: Erza sees Crime Sorciere off as they leave.
  Fact 0.54: Episode 155, 'Crocus, the Flower-Blooming Capital', has the Japanese title 'Hanasaku Miyako: Kurokkasu' and originally aired on November 3, 2012.
  Fact 0.55: Episode 155 aired in English on December 15, 2015.
  Fact 0.56: Natsu, Lucy, Gray, Erza, and Wendy are chosen to represent their guild in the Grand Magic Games.
  Fact 0.57: The Grand Magic Games are to be held in Fiore's capital, Crocus.
  Fact 0.58: As they tour the city, Natsu and Happy encounter the twin dragon slayers, Sting and Rogue.
  Fact 0.59: Sting and Rogue mock Natsu for not being able to kill Acnologia.
  Fact 0.60: Sting and Rogue boast of becoming true dragon slayers by killing the dragons who raised them.
  Fact 0.61: At midnight, Fairy Tail and the other participating guilds are put through a preliminary event.
  Fact 0.62: The preliminary event is a race through a massive floating labyrinth.
  Fact 0.63: The preliminary event is designed to narrow the number of participating guilds down to eight teams.
  Fact 0.64: Episode 156, 'Sky Labyrinth', has the Japanese title 'Sukai Rabirinsu' and originally aired on November 10, 2012.
  Fact 0.65: Episode 156 aired in English on December 22, 2015.
  Fact 0.66: Natsu's team notices that Wendy has gone missing.
  Fact 0.67: Elfman replaces Wendy to keep Natsu's team from being eliminated.
  Fact 0.68: As they make their way through the enchanted labyrinth, the five encounter various other guilds.
  Fact 0.69: The five steal the maps made by the other guilds to better navigate the maze.
  Fact 0.70: The five reach the end of the maze and pass the elimination round.
  Fact 0.71: The five are confident they have arrived first.
  Fact 0.72: The five discover they have barely taken last place.
  Fact 0.73: Happy, Lisanna, and the rest of the guild search for Wendy and Carla.
  Fact 0.74: Happy, Lisanna, and the rest of the guild find Wendy and Carla in the garden of the king's palace, Mercurius.
  Fact 0.75: Wendy and Carla have their magic power depleted.
  Fact 0.76: Episode 157, 'New Guild', has the Japanese title 'Shinki Girudo' and originally aired on November 17, 2012.
  Fact 0.77: Episode 157 aired in English on December 29, 2015.
  Fact 0.78: Wendy's body is weakened as a result of losing her magic power.
  Fact 0.79: Poluchka helps Wendy recover.
  Fact 0.80: Wendy formally requests Elfman to take her place in the Grand Magic Games.
  Fact 0.81: Fairy Tail steps into the arena with a negative audience reception.
  Fact 0.82: Fairy Tail is bolstered by their guild mates and the spirit of Mavis, whom only Fairy Tail guild members can see.
  Fact 0.83: The next four participating teams are Quatro Cerberus, Mermaid Heel, Blue Pegasus, and Lamia Scale.
  Fact 0.84: Raven Tail is a former dark guild led by Makarov's son Ivan.
  Fact 0.85: Raven Tail members confess to attacking Wendy and Carla.
  Fact 0.86: The next team is a second team from Fairy Tail composed of Gajeel, Juvia, Mirajane, Laxus, and Jellal.
  Fact 0.87: Jellal is disguised as Mystogan to help investigate the arena.
  Fact 0.88: The final guild to be announced is Saber Tooth.
  Fact 0.89: The audience cheers joyfully for Saber Tooth's announcement.
  Fact 0.90: All eight teams are announced.
  Fact 0.91: Each team sends out one member to participate in the first contest of the games.
  Fact 0.92: Gray and Juvia represent their teams in the first contest.
  Fact 0.93: Episode 158, 'Night of Shooting Stars', has the Japanese title 'Hoshi Furu Yoru ni' and originally aired on November 24, 2012.
  Fact 0.94: Episode 158 aired in English on January 5, 2016.
  Fact 0.95: The games' first contest, 'Hidden', requires the eight participants to score points by finding and attacking each other.
  Fact 0.96: The contest takes place in a virtual city filled with copies of the participants.
  Fact 0.97: Raven Tail member Nalpudding repeatedly targets Gray.
  Fact 0.98: Gray scores last place in the contest.
  Fact 0.99: Saber Tooth's Rufus scores first by using his Memory-Make magic to attack all the other participants at once.
  Fact 0.100: Gray is humiliated and swears revenge against Raven Tail and Saber Tooth.
  Fact 0.101: Lucy is chosen to fight Raven Tail member Flare in the tournament's first battle round.
  Fact 0.102: Episode 159, 'Lucy vs. Flare', has the Japanese title 'Rushii vs. Furea' and originally aired on December 1, 2012.
  Fact 0.103: Episode 159 aired in English on January 12, 2016.
  Fact 0.104: Lucy displays tremendous skill in her battle against Flare by summoning two spirits simultaneously.
  Fact 0.105: Flare cheats by using her magic hair to hold Alzack and Bisca's daughter, Asuka, as a hostage.
  Fact 0.106: Flare blackmails Lucy into not fighting back to keep Flare from harming Asuka.
  Fact 0.107: Lucy tearfully speaks of Asuka's danger.
  Fact 0.108: Natsu hears Lucy and rushes to Asuka.
  Fact 0.109: Natsu incinerates Flare's hair.
  Fact 0.110: Lucy counterattacks with her Urano Metoria spell in combination with Gemini.
  Fact 0.111: Flare's teammate Obra nullifies Lucy's magic before the spell fully activates.
  Fact 0.112: Lucy collapses and loses the match.
  Fact 0.113: Episode 160, 'Portent', has the Japanese title 'Kyozui' and originally aired on December 8, 2012.
  Fact 0.114: Episode 160 aired in English on January 19, 2016.
  Fact 0.115: Blue Pegasus' Ren defeats Mermaid Heel's Arana during the games' second match.
  Fact 0.116: Saber Tooth's Orga effortlessly defeats Quatro Cerberus' War Cry in the third match.
  Fact 0.117: For the day's fourth and final match, Jellal is pitted against Lamia Scale's Jura.
  Fact 0.118: Jellal intends to avenge the humiliated Fairy Tail members.
  Fact 0.119: Jellal mimics Mystogan's magic to keep up his disguise.
  Fact 0.120: Jellal finds Mystogan's magic ineffective against Jura.
  Fact 0.121: Jellal is forced to use his own magic instead.
  Fact 0.122: Before Jellal can cast a spell that will expose his true identity, Ultear uses Merudy's sensory to force Jellal into submission.
  Fact 0.123: The hospitalized Carla awakens after having a troubling premonition of the future.
  Fact 0.124: Episode 161, 'Chariots', has the Japanese title 'Chariotto' and originally aired on December 15, 2012.
  Fact 0.125: Episode 161 aired in English on January 26, 2016.
  Fact 0.126: Fairy Tail is undaunted by their poor performance on the first day.
  Fact 0.127: As the guild celebrates, they are visited and taunted by Bacchus.
  Fact 0.128: Bacchus is Erza's rival from Quatro Cerberus.
  Fact 0.129: Bacchus has come to replace War Cry.
  Fact 0.130: Carla relays her premonition to Poluchka.
  Fact 0.131: In Carla's premonition, Mercurius is collapsing around what appeared to be a tearfully singing Lucy.
  Fact 0.132: The second day of the tournament begins with a footrace atop a train of moving chariots.
  Fact 0.133: Natsu, Gajeel, and Sting suffer from motion sickness as a side effect of their Dragon Slayer magic.
  Fact 0.134: Natsu responds to Sting that his determination is a result of the strength of Fairy Tail's bonds.
  Fact 0.135: Sting eventually throws the race.
  Fact 0.136: Natsu finishes as sixth place.
  Fact 0.137: Gajeel finishes as seventh place.
  Fact 0.138: Natsu and Gajeel earn Fairy Tail the audience's respect.
  Fact 0.139: Episode 162, 'Elfman vs. Bacchus', has the Japanese title 'Erufuman vs. Bakkasu' and originally aired on December 22, 2012.
  Fact 0.140: Episode 162 aired in English on September 1, 2015.
  Fact 0.141: Due to a miscommunication between the king and his knight Arcadios, Elfman is chosen as Bacchus' opponent instead of Erza.
  Fact 0.142: Before the match, Bacchus and Elfman make a wager.
  Fact 0.143: If Bacchus wins, he will court Elfman's sisters.
  Fact 0.144: If Elfman wins, Bacchus' team name will be changed to 'Quatro Puppy' for the rest of the games.
  Fact 0.145: Bacchus ruthlessly attacks Elfman with his bare palms.
  Fact 0.146: Elfman damages Bacchus with his defensive Lizardman transformation.
  Fact 0.147: Elfman outlasts Bacchus as Bacchus exhausts himself.
  Fact 0.148: Elfman wins the match.
  Fact 0.149: Bacchus' team is renamed.
  Fact 0.150: Natsu finds Wendy, Carla, and Poluchka being kidnapped by a group of masked men.
  Fact 0.151: The masked men claim they were hired to kidnap Lucy by Raven Tail.
  Fact 0.152: The masked men are revealed to have actually been sent by Arcadios.
  Fact 0.153: Arcadios intends to sow discord between the guilds.
  Fact 0.154: Arcadios plots to use Lucy for his 'Eclipse Project'.
  Fact 0.155: Episode 163, 'Mirajane vs. Jenny', has the Japanese title 'Mirajen vs. Jeni' and originally aired on January 5, 2013.
  Fact 0.156: Episode 163 aired in English on September 1, 2015.
  Fact 0.157: The second day of the Grand Magic Games continues.
  Fact 0.158: Mirajane of Fairy Tail Team B fights against Jenny Realight of Team Blue Pegasus.
  Fact 0.159: Mirajane and Jenny are well-known for being magazine models.
  Fact 0.160: Their battle consists almost entirely of a fashion showdown.
  Fact 0.161: Jenny proposes a wager that the loser must pose nude for the Sorcerer Magazine.
  Fact 0.162: Mirajane accepts the wager.
  Fact 0.163: Jenny transforms into her battle armor.
  Fact 0.164: Mirajane shifts into her Satan Soul 'Sitri' form.
  Fact 0.165: Mirajane easily defeats Jenny.
  Fact 0.166: The final match of the day is Kagura Mikazuchi of Mermaid Heel against Yukino Aguria of Saber Tooth.
  Fact 0.167: Yukino proposes to Kagura that they wager their lives on their battle.
  Fact 0.168: Arcadios is confronted by Darton, the kingdom's defense minister, about his 'Eclipse Project'.
  Fact 0.169: Episode 164, 'Kagura vs. Yukino', has the Japanese title 'Kagura vs. Yukino' and originally aired on January 12, 2013.
  Fact 0.170: Episode 164 aired in English on September 1, 2015.
  Fact 0.171: Yukino is revealed to be a celestial spirit wizard when she summons Libra and Pisces to attack Kagura.
  Fact 0.172: Kagura defeats Libra and Pisces using her gravity-altering magic.
  Fact 0.173: Yukino uses a Gatekey that summons Ophiuchus, which is regarded as the strongest among the Celestial Spirits.
  Fact 0.174: Kagura slays Ophiuchus with her sword still sheathed.
  Fact 0.175: Kagura easily defeats Yukino.
  Fact 0.176: Jellal discusses the fact that he has not sensed the evil power similar to Zeref's.
  Fact 0.177: Jellal's discussion takes place after the second day of the Grand Magic Games has ended.
  Fact 0.178: Erza reunites with Milliana.
  Fact 0.179: Milliana is revealed to be a member of Mermaid Heel.
  Fact 0.180: Saber Tooth's master Jiemma is dissatisfied with his guild's performance.
  Fact 0.181: Sting is given another chance.
  Fact 0.182: Jiemma humiliates Yukino for her failure.
  Fact 0.183: Episode 165, 'Hatred at Nightfall', has the Japanese title 'Urami wa Yoru no Tobari ni Tsutsumarete' and originally aired on January 19, 2013.
  Fact 0.184: Episode 165 aired in English on September 1, 2015.
  Fact 0.185: After Yukino is expelled from Saber Tooth, she offers her two golden Gatekeys to Lucy.
  Fact 0.186: Yukino offers the Gatekeys to complete Lucy's collection of the Zodiac Gatekeys.
  Fact 0.187: Yukino states that when the twelve Zodiac Gatekeys are all gathered, a gateway that changes the world will open.
  Fact 0.188: Lucy turns down the offer.
  Fact 0.189: Milliana tells Erza that Kagura's sword, Archenemy, is intended to kill Jellal.
  Fact 0.190: Milliana tells Erza that she and Kagura share the same resentment towards Jellal.
  Fact 0.191: Natsu and Happy run after Yukino in an attempt to apologize.
  Fact 0.192: Natsu realizes that Yukino had been expelled.
  Fact 0.193: Natsu infiltrates the Crocus Garden to fight Jiemma.
  Fact 0.194: Episode 166, 'Pandemonium', has the Japanese title 'Pandemoniumu' and originally aired on January 26, 2013.
  Fact 0.195: Episode 166 aired in English on September 1, 2015.
  Fact 0.196: Natsu punches out Dobengal, the guild member summoned to fight him.
  Fact 0.197: Natsu and Jiemma engage in a duel.
  Fact 0.198: The guild master's daughter, Minerva, appears and ends their duel.
  Fact 0.199: Minerva takes Happy hostage.
  Fact 0.200: Minerva convinces Natsu to settle their dispute in the Grand Magic Games.
  Fact 0.201: As Natsu leaves, he states that Saber Tooth will never surpass Fairy Tail because they care for their friends.
  Fact 0.202: Rogue echoes Natsu's words.
  Fact 0.203: Sting is left in awe at Natsu's strength.
  Fact 0.204: The following day, an event called 'Pandemonium' begins.
  Fact 0.205: In Pandemonium, participants enter a magical citadel filled with 100 monsters of varying strengths.
  Fact 0.206: Participants individually do battle with any number of monsters they choose.
  Fact 0.207: Erza, representing Fairy Tail Team A, requests the 100 monsters as her opponents.
  Fact 0.208: Episode 167, '100 Against 1', has the Japanese title '100 tai 1' and originally aired on February 2, 2013.
  Fact 0.209: Episode 167 aired in English on September 1, 2015.
  Fact 0.210: Erza kills the 100 monsters using various types of armor to match their strengths and weaknesses.
  Fact 0.211: The crowd praises Fairy Tail.
  Fact 0.212: Fairy Tail is awarded 10 points.
  Fact 0.213: The game officials decide to determine the ranks of the remaining participants by means of a device called the 'Magic Power Finder', sometimes abbreviated as MPF.
  Fact 0.214: Participants must attack the Magic Power Finder with their magic power.
  Fact 0.215: The Magic Power Finder calculates and displays participants' equivalent scores based on the extent of power used.
  Fact 0.216: Cana of Fairy Tail Team B dominates the event by using the Fairy Glitter spell lent to her by Mavis.
  Fact 0.217: Makarov is surprised by Cana's use of the Fairy Glitter spell.
  Fact 0.218: Fairy Tail Team B receives 8 points.
  Fact 0.219: The audience bursts into a resounding swarm of cheers and claps due to the finish of the two Fairy Tail teams.
  Fact 0.220: The masked leader Alexei tells his subordinates Flare, Obra, Nalpudding, and Kurohebi that it is about time to proceed with their true mission.
  Fact 0.221: Raven Tail somehow got insider information regarding the battle matchups for the day.
  Fact 0.222: Episode 168, 'Laxus vs. Alexei', has the Japanese title 'Rakusasu vs. Arekusei' and originally aired on February 9, 2013.
  Fact 0.223: Episode 168 aired in English on September 1, 2015.
  Fact 0.224: In the battle portion of the Grand Magic Games' third day, Milliana of Mermaid Heel beats Semmes of Quatro Puppy.
  Fact 0.225: Saber Tooth's Rufus swiftly defeats Blue Pegasus' Eve.
  Fact 0.226: Laxus of Fairy Tail Team B is chosen to battle against Alexei of Team Raven Tail.
  Fact 0.227: The Fairy Tail members take extra precautions because Raven Tail is a suspicious guild.
  Fact 0.228: Fairy Tail watches the actions of Ivan Dreyar and the other four Raven Tail members.
  Fact 0.229: The spectators are shocked to see Laxus being beaten by Alexei.
  Fact 0.230: No cautious movements can be seen from the Raven Tail team.
  Fact 0.231: The two dueling people are illusions created by Alexei.
  Fact 0.232: Alexei created the illusions so that he could have a talk with Laxus.
  Fact 0.233: The other Raven Tail members being monitored by Fairy Tail are lifelike illusions.
  Fact 0.234: In reality, the other Raven Tail members are with Alexei on the battlefield.
  Fact 0.235: Alexei reveals himself to be Ivan Dreyar.
  Fact 0.236: Ivan and the Raven Tail members are unable to interrogate Laxus about the Lumen Histoire's location.
  Fact 0.237: Ivan and the Raven Tail members are forced to fight Laxus.
  Fact 0.238: Before their battle begins, Ivan declares that Raven Tail is an anti-Fairy Tail guild.
  Fact 0.239: Episode 169, 'Wendy vs. Shelia', has the Japanese title 'Wendi vs. Sheria' and originally aired on February 16, 2013.
  Fact 0.240: Episode 169 aired in English on September 1, 2015.
  Fact 0.241: Laxus defeats Ivan and the four Raven Tail elite members.
  Fact 0.242: Ivan's illusions disappear to the shock of everyone in the stadium.
  Fact 0.243: The game officials declare Laxus the victor.
  Fact 0.244: Ivan tells Laxus that Lumen Histoire is Fairy Tail's 'darkness'.
  Fact 0.245: Raven Tail is disqualified for the remainder of the Games due to the infraction of the rules.
  Fact 0.246: Raven Tail's members are captured by the soldiers.
  Fact 0.247: Wendy of Fairy Tail Team A and Sherria Blendy of Team Lamia Scale face off for the final match of the day.
  Fact 0.248: Sherria is revealed to be a Sky God Slayer.
  Fact 0.249: Gusts of wind surround the arena as Sherria and Wendy exchange attacks.
  Fact 0.250: Wendy employs an advanced Sky Dragon Slayer attack, much to Poluchka's astonishment.
  Fact 0.251: Sherria falls on the ground.
  Fact 0.252: Wendy is about to be announced as the winner of the bout.
  Fact 0.253: Sherria suddenly stands up with her wounds fully healed.
  Fact 0.254: The Crime Sorciere guild senses the nefarious aura.
  Fact 0.255: Jellal proceeds to the Domus Flau audience.
  Fact 0.256: Jellal is oblivious that he bumped into Doranbolt.
  Fact 0.257: Doranbolt wonders what Mystogan is doing in Earthland.
  Fact 0.258: Episode 170, 'Small Fists', has the Japanese title 'Chiisana Kobushi' and originally aired on February 23, 2013.
  Fact 0.259: Episode 170 aired in English on September 1, 2015.
  Fact 0.260: Cheria is fully healed.
  Fact 0.261: Wendy is almost depleted of magic power.
  Fact 0.262: Cheria casts a high-level spell towards Wendy.
  Fact 0.263: Wendy enhances Cheria's physical strength, causing the attack to miss Wendy.
  Fact 0.264: The two competitors engage in hand-to-hand combat.
  Fact 0.265: Due to the time limit and neither side giving up, their battle results in a draw.
  Fact 0.266: The draw gives both teams 5 points.
  Fact 0.267: Jellal tracks the sinister aura among the crowd in the form of a mysterious, hooded person.
  Fact 0.268: Jellal is about to confront the aura source.
  Fact 0.269: Doranbolt, Lahar, and other Rune Knights corner Jellal.
  Fact 0.270: Doranbolt interrogates Jellal.
  Fact 0.271: Jellal's cover is blown.
  Fact 0.272: The Rune Knights, Kagura and Milliana see the true identity of Mystogan.
  Fact 0.273: Yajima convinces the Rune Knights that Mystogan is Jellal's Edolas counterpart.
  Fact 0.274: Lahar releases the fugitive.
  Fact 0.275: Laxus and Makarov have a small conversation regarding Lumen Histoire.
  Fact 0.276: Mavis appears and explains that Lumen Histoire is the 'light' of Fairy Tail.
  Fact 0.277: Episode 171, 'Naval Battle', has the Japanese title 'Nabaru Batoru' and originally aired on March 2, 2013.
  Fact 0.278: Episode 171 aired in English on September 1, 2015.
  Fact 0.279: The jubilant Fairy Tail members celebrate their victories.
  Fact 0.280: Sting, Kagura, and Milliana are contemplating their own concerns.
  Fact 0.281: Yukino is enlisted as a sergeant under Arcadios' command.
  Fact 0.282: The fourth day of the competitions begins with an event called 'Naval Battle'.
  Fact 0.283: Naval Battle involves participants swimming in a giant water sphere and knocking the other competitors out.
  Fact 0.284: After the other participants fall out of the sphere, a five-minute battle between Lucy and Minerva ensues.
  Fact 0.285: Minerva quickly attacks Lucy and steals the keys of the Celestial Spirit mage.
  Fact 0.286: Lucy refuses to surrender.
  Fact 0.287: After hearing Lucy's intentions, Minerva stops attacking until the five-minute mark is reached.
  Fact 0.288: Minerva resumes attacking Lucy until the game officials decide to halt the match.
  Fact 0.289: The Fairy Tail members snap.
  Fact 0.290: Minerva dangles Lucy out of the sphere.
  Fact 0.291: The other Fairy Tail members rush to Lucy's aid.
  Fact 0.292: Episode 172, 'A Parfum for You', has the Japanese title 'Kimi ni Sasageru Parufamu' and originally aired on March 9, 2013.
  Fact 0.293: Episode 172 aired in English on September 1, 2015.
  Fact 0.294: Lucy is hospitalized and taken to the infirmary.
  Fact 0.295: The Fairy Tail members warn Saber Tooth that they have angered the worst possible guild to do so.
  Fact 0.296: Makarov announces that the two Fairy Tail teams must merge.
  Fact 0.297: The merger is a result of the disqualification of Raven Tail.
  Fact 0.298: The merger compensates for the odd number of teams for the battle round.
  Fact 0.299: The new team will have the former Team A's 35 points to start with.
  Fact 0.300: The new Fairy Tail team consists of Natsu, Gray, Erza, Gajeel, and Laxus.
  Fact 0.301: The new team is well-received as they enter the field.
  Fact 0.302: The second part of the tournament begins with a tag battle between Blue Pegasus' Ichiya and 'Rabbit' and Quatro Puppy's Rocker and Bacchus.
  Fact 0.303: 'Rabbit' is revealed to be Ichiya in disguise.
  Fact 0.304: Bacchus punches the Exceed.
  Fact 0.305: Ichiya bulks up to defeat the Quatro Puppy opponents.
  Fact 0.306: Blue Pegasus is awarded 10 points.
  Fact 0.307: Episode 173, 'Battle of Dragon Slayers!', has the Japanese title 'Batoru obu Doragon Sureiya' and originally aired on March 16, 2013.
  Fact 0.308: Episode 173 aired in English on September 1, 2015.
  Fact 0.309: Mermaid Heel's Kagura and Milliana face off against Lamia Scale's Lyon and Yuka for the second tag battle of the games.
  Fact 0.310: The second tag battle ends in a draw due to the time limit.
  Fact 0.311: For the third and final match of the day, Natsu and Gajeel of Team Fairy Tail enter the field against Sting and Rogue of Team Saber Tooth.
  Fact 0.312: Sting and Rogue were about to take the initiative when Natsu and Gajeel attacked them.
  Fact 0.313: The audience watches in shock as the Fairy Tail Dragon Slayers overpower the Saber Tooth Dragon Slayers.
  Fact 0.314: Sting activates 'White Drive'.
  Fact 0.315: Rogue powers up using 'Shadow Drive'.
  Fact 0.316: Episode 174, 'Four Dragons', has the Japanese title 'Yonin no Doragon' and originally aired on March 23, 2013.
  Fact 0.317: Episode 174 aired in English on September 1, 2015.
  Fact 0.318: Sting and Rogue begin attacking Natsu and Gajeel with their powers increased.
  Fact 0.319: After receiving heavy blows from the Saber Tooth Dragon Slayers, Natsu and Gajeel manage to hold them off.
  Fact 0.320: Sting and Rogue are forced to activate 'Dragon Force' on their own will.
  Fact 0.321: The Fairy Tail audience is shocked by Sting and Rogue activating 'Dragon Force'.
  Fact 0.322: Sting decides to finish the Fairy Tail Dragon Slayers alone.
  Fact 0.323: Sting held a promise to his Exceed friend Lector many years ago.
  Fact 0.324: The tide turns in favor of Sting as he unleashes a powerful attack on Natsu and Gajeel.
  Fact 0.325: Sting's attack causes the collapse of the arena floor.
  Fact 0.326: After a hand-to-hand combat underground, Sting is seemingly the victor of the match.
  Fact 0.327: Natsu and Gajeel stand up, comment on Sting's strength, and argue about his position while attacking.
  Fact 0.328: Natsu drags Gajeel on a mining cart and sends it away.
  Fact 0.329: Gajeel is left incapacitated due to motion sickness.
  Fact 0.330: Natsu challenges the Twin Dragons of Saber Tooth alone, in retribution for what Sting did earlier.
  Fact 0.331: Episode 175, 'Natsu vs. the Twin Dragons', has the Japanese title 'Natsu vs. Soryu' and originally aired on March 30, 2013.
  Fact 0.332: Episode 175 aired in English on September 1, 2015.
  Fact 0.333: The tag battle of the Dragon Slayers comes to a close.
  Fact 0.334: Natsu overpowers the combined attacks of Sting and Rogue.
  Fact 0.335: The Saber Tooth members watch in dismay.
  Fact 0.336: Jiemma seethes in anger over the defeat of the Twin Dragons.
  Fact 0.337: Jiemma's anger frightens the nearby spectators away.
  Fact 0.338: The rest of Fairy Tail celebrates in joy as its team receives 10 points.
  Fact 0.339: Fairy Tail surpasses Team Saber Tooth by a single point.
  Fact 0.340: Gajeel, back to his senses after being dragged by Natsu into a cart earlier, stumbles upon a cave full of dragon skeletons.
  Fact 0.341: That night, Jellal manages to corner the source of the dark aura.
  Fact 0.342: Jellal stands in disbelief at the identity of the mysterious, hooded person.

Retrieved items:
<item 1>
List of Fairy Tail episodes (season 6)
The following day as the Grand Magic Games continue , an event called `` Pandemonium '' begins . It involves the participants entering a magical citadel filled
with 100 monsters of varying strengths and individually doing battle with any number of monsters they choose . Representing Fairy Tail Team A , Erza requests
the 100 monsters as her opponents . 167 `` 100 Against 1 '' `` Hyaku tai Ichi '' ( 100 対 1 ) February 2 , 2013 September 1 , 2015 Erza kills the 100 monsters
using various types of armor to match their strength and weaknesses . As a result , the crowd praises Fairy Tail , and the guild is awarded with 10 points .
Moreover , the game officials decided to determine the ranks of the remaining participants by means of a device called `` Magic Power Finder '' ( sometimes
abbreviated as MPF ) . Using this tool , participants must attack it with their magic power , which will , in turn , calculate and display their equivalent
scores depending on the extent of power used .
</item 1>

<item 2>
List of Fairy Tail episodes (season 6)
Later , Jellal discusses the fact that he has not sensed the evil power similar to Zeref 's even though the second day of the Grand Magic Games has ended . Erza
reunites with Milliana , who is revealed to be a member of Mermaid Heel . Meanwhile , Saber Tooth 's master Jiemma is dissatisfied with his guild 's performance
. Though Sting is given another chance , Jiemma humiliates Yukino for her failure . 165 `` Hatred at Nightfall '' `` Urami wa Yoru no Tobari ni Tsutsumarete ''
( 怨み は 夜 の 帐 に 包 まれ て ) January 19 , 2013 September 1 , 2015 After Yukino is expelled from Saber Tooth , she offers her two golden Gatekeys to Lucy to complete
the latter 's collection of the Zodiac Gatekeys , stating that when the twelve Zodiac Gatekeys are all gathered , a gateway that changes the world will open ;
however , Lucy turns down the offer .
</item 2>

<item 3>
List of Fairy Tail episodes (season 6)
To everyone 's shock , the next team is revealed to be a second team from Fairy Tail composed of Gajeel , Juvia , Mirajane , Laxus , and Jellal ( disguised as
Mystogan to help investigate the arena ) . The final guild to be announced is Saber Tooth , much to the joy of the audience . With all eight teams announced ,
they each send out one member to participate in the first contest of the games , with Gray and Juvia representing their teams . 158 `` Night of Shooting Stars
'' `` Hoshi Furu Yoru ni '' ( 星 降 ル 夜 ニ ) November 24 , 2012 January 5 , 2016 The games ' first contest , `` Hidden '' , requires the eight participants to
score points by finding and attacking each other in a virtual city filled with copies of themselves . Raven Tail member Nalpudding repeatedly targets Gray , who
scores last place in the contest while Saber Tooth 's Rufus scores first by using his Memory - Make magic to attack all the other participants at once .
</item 3>

<item 4>
List of Fairy Tail episodes (season 6)
In the end , Cana of Fairy Tail Team B dominates the event by using the Fairy Glitter spell lent to her by Mavis , much to Makarov 's surprise . With Cana 's
victory , Fairy Tail Team B receives 8 points , and the audience began to burst once more into a resounding swarm of cheers and claps due to the finish of the
two Fairy Tail teams . Meanwhile , the Raven Tail team somehow managed to get an insider information regarding the battle matchups for the day , so the masked
leader Alexei tells his subordinates Flare , Obra , Nalpudding , and Kurohebi that it 's about time to proceed with their true mission . 168 `` Laxus vs. Alexei
'' `` Rakusasu vs. Arekusei '' ( ラクサス vs. アレクセイ ) February 9 , 2013 September 1 , 2015 In the battle portion of the Grand Magic Games ' third day , Milliana of
Mermaid Heel beats Semmes of Quatro Puppy while Saber Tooth 's Rufus swiftly defeats Blue Pegasus ' Eve .
</item 4>

<item 5>
List of Fairy Tail episodes (season 6)
When the wizards arrive , they instead find a surprise party held by the Celestial Spirit King celebrating their return from Sirius Island after seven years .
They spend the day in the spirit world partying with the spirits . Before they leave , however , Virgo informs them that one day in the spirit world equals
three months in the human world , and they find the Grand Magic Games are upon them . 154 `` For All the Time We Missed Each Other '' `` Surechigatta Jikan no
Bun dake '' ( すれ 违 っ た 时间 の 分 だけ ) October 27 , 2012 December 8 , 2015 While mulling over their lost training time , the Fairy Tail wizards receive a message
via carrier pigeon requesting a private meeting in the forest nearby . There they meet a runaway Jellal , Ultear , and Merudy , who have formed their own guild
, Crime Sorcière , to defeat Zeref and any dark guilds that try to use his power .
</item 5>

<item 6>
Graph query row:
event: Grand Magic Games
start_date_or_timing: in three months time
connected_to: Fiore
relationship: LOCATED_IN
</item 6>

<item 7>
List of Fairy Tail episodes (season 6)
Poluchka reveals herself as the Edolas counterpart of Grandine , and gives Wendy written instructions for two advanced spells that Grandine failed to teach her
. Meanwhile , Makarov announces Gildarts ' succession as master to the rest of the guild , only to find he has gone on a journey , leaving a letter of
resignation behind . In the letter , Gildarts reinstates Laxus into the guild , returns mastership to Makarov , and tells him to make Fairy Tail the strongest
guild in Fiore again . Romeo suggests they participate in the Grand Magic Games , an annual festival designed to determine the guild rankings in Fiore . 153 ``
Song of the Stars '' `` Hoshiboshi no Uta '' ( 星々 の 歌 ) October 20 , 2012 December 1 , 2015 With the Grand Magic Games occurring in three months time , Natsu
and his friends train at a beach resort . On their second day of training , Virgo appears and tells the wizards that the Celestial Spirit World is in danger of
destruction .
</item 7>

<item 8>
List of Fairy Tail episodes (season 6)
The three ask Fairy Tail for their help in determining the source of a magical anomaly similar to Zeref 's magic that they have sensed around the Grand Magic
Games arena , Domus Flau . In exchange , Ultear uses her magic to invoke their Second Origin in order to double their magic power . As the others undergo the
painful process of awakening their Second Origin , Erza talks with Jellal over his regained memories . The two nearly kiss , but Jellal decides that he and Erza
can not be together and lies to her that he has a fiancée . Erza quickly picks up on this and respects his decision , seeing Crime Sorcière off as they leave .
155 `` Crocus , the Flower - Blooming Capital '' `` Hanasaku Miyako : Kurokkasu '' ( 花 笑 く 都 ・ クロッカス ) November 3 , 2012 December 15 , 2015 Natsu , Lucy , Gray
, Erza , and Wendy are chosen to represent their guild in the Grand Magic Games , which are to be held in Fiore 's capital , Crocus .
</item 8>

<item 9>
List of Fairy Tail episodes (season 6)
It contains a single story arc , titled `` Dai Matō Enbu '' ( 大 魔 闘 演武 , lit `` Grand Magic Games '' ) , which adapts material from the beginning of the 31st
through the ending of the 35th volumes of Hiro Mashima 's Fairy Tail manga over 24 episodes . Focusing on Natsu and the others who were frozen in time for seven
years on Tenrou Island , the members return to Magnolia and discover that Saber Tooth is the strongest guild in the Fiore Kingdom . Having become the weakest
guild during the absence of its core members , Fairy Tail decides to enter the Grand Magic Games , an annual competition to decide Fiore 's strongest guild .
The season began airing on October 6 , 2012 on TV Tokyo in Japan and concluded its run on March 30 , 2013 . The first DVD compilation , containing the last two
episodes of the fifth season and the first two episodes of the sixth season , was released by Pony Canyon on March 6 , 2013 .
</item 9>

<item 10>
List of Fairy Tail episodes (season 6)
Meanwhile , Natsu finds Wendy , Carla , and Poluchka being kidnapped by a group of masked men . Upon being caught , the men claim they were hired to kidnap Lucy
by Raven Tail . They are soon revealed to have actually been sent by Arcadios , who intends to sow discord between the guilds as he plots to use Lucy for his ``
Eclipse Project '' . 163 `` Mirajane vs. Jenny '' `` Mirajēn vs. Jenī '' ( ミラ ジェーン vs. ジェニー ) January 5 , 2013 September 1 , 2015 The second day of the Grand
Magic Games continues , with Mirajane of Fairy Tail Team B fighting against Jenny Realight of Team Blue Pegasus . Since the two competitors are well - known for
being magazine models , their battle consists almost entirely of a fashion showdown , much to the pleasure of the audience . Eventually as the showdown becomes
redundant , Jenny proposes a wager : the loser must pose nude for the Sorcerer Magazine , which Mirajane accepts .
</item 10>
=== END ITEM id=31f44c28761e3456 ===
