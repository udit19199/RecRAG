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

This file has 7 items.

=== ITEM id=2b63298f615363b1 ===
Gold sentences and facts:
Sentence 0: Deion Sanders Sanders in 2011 No. 21 , 37 Position : Cornerback Date of birth : ( 1967 - 08 - 09 ) August 9 , 1967 ( age 50 ) Place of birth : Fort Myers , Florida Height : 6 ft 1 in ( 1.85 m ) Weight : 198 lb ( 90 kg ) Career information High school : North Fort Myers ( FL ) College : Florida State NFL Draft : 1989 / Round : 1 / Pick : 5 Career history Atlanta Falcons ( 1989 -- 1993 ) San Francisco 49ers ( 1994 ) Dallas Cowboys ( 1995 -- 1999 ) Washington Redskins ( 2000 ) Baltimore Ravens ( 2004 -- 2005 ) Career highlights and awards 2 × Super Bowl champion ( XXIX , XXX ) 8 × Pro Bowl ( 1991 -- 1994 , 1996 -- 1999 ) 8 × First - team All - Pro ( 1991 -- 1994 , 1996 -- 1999 ) NFL Defensive Player of the Year ( 1994 ) NFL 1990s All - Decade Team Atlanta Falcons Ring of Honor Jim Thorpe Award ( 1988 ) 2 × Unanimous All - American ( 1987 , 1988 ) Florida State Seminoles No. 2 retired Career NFL statistics Interceptions : 53 Defensive touchdowns : 10 Return touchdowns : 9 Receptions : 60 Receiving yards : 784 Receiving touchdowns : 3 Player stats at NFL.com Player stats at PFR Pro Football Hall of Fame College Football Hall of Fame
  Fact 0.0: A 2011 photo of Deion Sanders is captioned No. 21, 37.
  Fact 0.1: Deion Sanders played the position of cornerback.
  Fact 0.2: Deion Sanders was born on August 9, 1967.
  Fact 0.3: Deion Sanders is listed as age 50.
  Fact 0.4: Deion Sanders was born in Fort Myers, Florida.
  Fact 0.5: Deion Sanders is 6 ft 1 in (1.85 m) tall.
  Fact 0.6: Deion Sanders weighs 198 lb (90 kg).
  Fact 0.7: Deion Sanders attended North Fort Myers High School in Florida.
  Fact 0.8: Deion Sanders attended Florida State University for college.
  Fact 0.9: Deion Sanders was selected in round 1, pick 5 of the 1989 NFL Draft.
  Fact 0.10: Deion Sanders played for the Atlanta Falcons from 1989 to 1993.
  Fact 0.11: Deion Sanders played for the San Francisco 49ers in 1994.
  Fact 0.12: Deion Sanders played for the Dallas Cowboys from 1995 to 1999.
  Fact 0.13: Deion Sanders played for the Washington Redskins in 2000.
  Fact 0.14: Deion Sanders played for the Baltimore Ravens from 2004 to 2005.
  Fact 0.15: Deion Sanders won 2 Super Bowls, Super Bowl XXIX and Super Bowl XXX.
  Fact 0.16: Deion Sanders was selected to the Pro Bowl 8 times, in 1991 to 1994 and 1996 to 1999.
  Fact 0.17: Deion Sanders was named First-team All-Pro 8 times, in 1991 to 1994 and 1996 to 1999.
  Fact 0.18: Deion Sanders was named NFL Defensive Player of the Year in 1994.
  Fact 0.19: Deion Sanders was named to the NFL 1990s All-Decade Team.
  Fact 0.20: Deion Sanders is a member of the Atlanta Falcons Ring of Honor.
  Fact 0.21: Deion Sanders won the Jim Thorpe Award in 1988.
  Fact 0.22: Deion Sanders was a 2-time Unanimous All-American, in 1987 and 1988.
  Fact 0.23: Florida State Seminoles retired No. 2.
  Fact 0.24: Deion Sanders had 53 interceptions in his NFL career.
  Fact 0.25: Deion Sanders had 10 defensive touchdowns in his NFL career.
  Fact 0.26: Deion Sanders had 9 return touchdowns in his NFL career.
  Fact 0.27: Deion Sanders had 60 receptions in his NFL career.
  Fact 0.28: Deion Sanders had 784 receiving yards in his NFL career.
  Fact 0.29: Deion Sanders had 3 receiving touchdowns in his NFL career.
  Fact 0.30: Player stats for Deion Sanders are listed at NFL.com and Pro Football Reference.
  Fact 0.31: Deion Sanders is a member of the Pro Football Hall of Fame.
  Fact 0.32: Deion Sanders is a member of the College Football Hall of Fame.

Retrieved items:
<item 1>
Deion Sanders
1996 -- 1999 ) NFL Defensive Player of the Year ( 1994 ) NFL 1990s All - Decade Team Atlanta Falcons Ring of Honor Jim Thorpe Award ( 1988 ) 2 × Unanimous All -
American ( 1987 , 1988 ) Florida State Seminoles No. 2 retired Career NFL statistics Interceptions : 53 Defensive touchdowns : 10 Return touchdowns : 9
Receptions : 60 Receiving yards : 784 Receiving touchdowns : 3 Player stats at NFL.com Player stats at PFR Pro Football Hall of Fame College Football Hall of
Fame Deion Luywnn Sanders Sr. ( / ˈdiːɒn / ; born August 9 , 1967 ) , nicknamed Primetime , is a former American football and baseball player who works as an
analyst for CBS Sports and the NFL Network . He was inducted into the Pro Football Hall of Fame in 2011 . Sanders played football primarily at cornerback , but
also as a kick returner , punt returner and occasionally wide receiver .
</item 1>

<item 2>
Deion Sanders
Deion Sanders - wikipedia Deion Sanders Jump to : navigation , search Deion Sanders Sanders in 2011 No. 21 , 37 Position : Cornerback Date of birth : ( 1967 -
08 - 09 ) August 9 , 1967 ( age 50 ) Place of birth : Fort Myers , Florida Height : 6 ft 1 in ( 1.85 m ) Weight : 198 lb ( 90 kg ) Career information High
school : North Fort Myers ( FL ) College : Florida State NFL Draft : 1989 / Round : 1 / Pick : 5 Career history Atlanta Falcons ( 1989 -- 1993 ) San Francisco
49ers ( 1994 ) Dallas Cowboys ( 1995 -- 1999 ) Washington Redskins ( 2000 ) Baltimore Ravens ( 2004 -- 2005 ) Career highlights and awards 2 × Super Bowl
champion ( XXIX , XXX ) 8 × Pro Bowl ( 1991 -- 1994 , 1996 -- 1999 ) 8 × First - team All - Pro ( 1991 -- 1994 ,
</item 2>

<item 3>
Deion Sanders
On February 5 , 2011 , Sanders was announced as a Pro Football Hall of Fame inductee in his first year of eligibility . On May 17 , 2011 , Sanders was announced
as a College Football Hall of Fame inductee . On August 6 , 2011 , Sanders was inducted into the Pro Football Hall of Fame . Sanders also had a rushing TD in
the playoffs ( against the Philadelphia Eagles in January 1996 ) . This makes him ( including post season ) one of only two players in NFL history ( Bill Dudley
being the other ) to score a touchdown six different ways ( interception return , punt return , kickoff return , receiving , rushing , and a fumble recovery ) .
On February 6 , 2011 , at Super Bowl XLV in Arlington , Texas , Sanders performed the pre-game coin toss . Sanders did not attend classes or take final exams
during the fall semester of his senior year at Florida State , yet played in the Sugar Bowl against Auburn . This caused the Florida State Legislature to create
the ' Deion Sanders rule ' , whereby a football athlete at any state university could not play in a bowl game if he failed to successfully complete the previous
semester .
</item 3>

<item 4>
Deion Sanders
WR Wendell Davis TE Keith Jackson OL Mark Hutson OL Dave Cadigan OL John Elliott OL Randall McDaniel C Nacho Albergamo Defense DL Daniel Stubbs DL Chad Hennings
DL Tracy Rocker DL Ted Gregory DL John Roper LB Chris Spielman LB Aundray Bruce LB Dante Jones DB Bennie Blades DB Deion Sanders DB Rickey Dixon DB Chuck Cecil
Special teams PK David Treadwell P Tom Tupa 1988 College Football All - America Team consensus selections Offense QB Steve Walsh & Troy Aikman RB Barry Sanders
RB Anthony Thompson RB Tim Worley WR Jason Phillips WR Hart Lee Dykes TE Marv Cook OL Tony Mandarich OL Anthony Phillips OL Mike Utley OL Mark Stepnoski C Jake
Young C John Vitale Defense DL Mark Messner DL Tracy Rocker DL Wayne Martin DL Frank Stams DL Bill Hawkins LB Derrick Thomas LB Broderick Thomas LB Mike
Stonebreaker DB Deion Sanders DB Donnell Woolford DB Louis Oliver DB Darryl Henley Special teams PK Kendall Trainor P Keith English Jim Thorpe Award winners
1986 : Everett 1987 : Blades & Dixon 1988 : Sanders 1989 : Carrier 1990 : Lewis 1991 : Buckley
</item 4>

<item 5>
Deion Sanders
NFL . National Football League . Retrieved 28 January 2017 . Jump up ^ `` Deion Sanders HOF Speech '' . Video Entry . Retrieved October 19 , 2011 . Jump up ^
Kallestad , Brent . `` Deion Changes the Rules Again : Pro football : This time , however , it 's for better education of athletes with new standards in Florida
. '' . Los Angeles Times . Retrieved 29 January 2017 . Jump up ^ Trex , Ethan . `` Two Rules Named After Deion Sanders '' . mental_floss . Mental Floss , Inc .
Retrieved 29 January 2017 . Jump up ^ `` NFL Videos : 2011 HOF : Deion Sanders '' . Nfl.com . August 6 , 2011 . Retrieved August 16 , 2012 . Jump up ^ Frohman ,
Jimmy . `` 11 Most Awesome NFL Player Nicknames '' . Phactual . Phactual , LLC . Retrieved 24 January 2017 . Jump up ^ `` CES ProNews Flashes ! '' .
</item 5>

<item 6>
Deion Sanders
Sanders amassed 7,838 all - purpose yards and scored 22 touchdowns : nine interception returns , six punt returns , three kickoff returns , three receiving ,
and one fumble recovery . His 19 defensive and return touchdowns was an NFL record ( now held by Devin Hester ) . In the postseason , Sanders added 5 more
interceptions , as well as 3 receptions for 95 yards , 4 carries for 39 yards , and two touchdowns ( one rushing and one receiving ) . He was selected to eight
Pro Bowls and won the NFL Defensive Player of the Year Award in 1994 . College Football News named Sanders # 8 in its list of 100 Greatest College Football
Players of All - Time . The Sporting News named Sanders # 37 in their Top 100 Football Players of the Century released in 1999 . ESPN named Sanders # 74 in its
list of the 100 Great Athletes of the Century released in 1999 . NFL.com named Sanders # 34 on NFL 's Top 100 list released in late 2010 On November 11 , 2010 ,
Sanders was inducted into the Atlanta Falcons ' Ring of Honor .
</item 6>

<item 7>
Deion Sanders
Sanders also occasionally lined up with his team 's offense . During the 1996 season , Sanders skipped the baseball season , concentrating on football , and
attended the first NFL training camp of his career to better familiarize himself with the nuances of the wide receiver position . He became only the second two
- way starter ( after the Cardinals ' Roy Green ) in the NFL since Chuck Bednarik for the first half of the season due . Sanders is the only man to play in both
a Super Bowl and a World Series , to hit an MLB home run and score an NFL touchdown in the same week , and to have both a reception and an interception in the
Super Bowl . He is one of two players to score an NFL touchdown six different ways . During his career , Sanders intercepted 53 passes for 1,331 yards ( a 25.1
yards per return average ) , recovered four fumbles for 15 yards , returned 155 kickoffs for 3,523 yards , gained 2,199 yards on 212 punt returns , and caught
60 passes for 784 yards .
</item 7>

<item 8>
Deion Sanders
In 2014 , he debuted his new show , the Deion Family Playbook . Contents ( hide ) 1 Early life 1.1 High school 1.2 College career 2 Professional baseball career
3 Professional football career 3.1 Draft and Atlanta Falcons 3.2 San Francisco 49ers 3.3 Dallas Cowboys 3.4 Washington Redskins 3.5 Baltimore Ravens 4 Legacy 5
Media appearances and pop culture fame 6 Other business and entertainment ventures 7 Discography 8 Personal life 9 See also 10 References 11 External links
Early life ( edit ) High school ( edit ) Sanders was born in Fort Myers , Florida . He attended North Fort Myers High School in North Fort Myers , and was a
letterman and All - State honoree in football , basketball and baseball . In 1985 , Sanders was named to the Florida High School Association All - Century Team
which selected the top 33 players in the 100 - year history of high school football in the state . The Kansas City Royals selected Sanders out of North Fort
Myers High School in the sixth round of the 1985 Major League Baseball draft . He did not sign with the Royals .
</item 8>

<item 9>
Deion Sanders
In 1995 , he signed with the Dallas Cowboys for a minimum yearly base salary and a nearly $13 million signing bonus in an attempt to circumvent the NFL 's
salary cap . This caused the NFL to institute its own ' Deion Sanders rule ' whereby a prorated portion of a player 's signing bonus counted against the salary
cap . Media appearances and pop culture Fame ( edit ) See also : Leon Sandcastle Sanders became known for sporting a `` do - rag '' or bandana and for his high
- stepping into the end zone followed by his touchdown dance celebrations . At the end of his Hall of Fame speech , he put a bandana on his bust . His `` Prime
Time '' nickname was given to him by a friend and high - school teammate , Florida Gators defensive back Richard Fain . The two played pickup basketball games
together during the prime time television hour , and Sanders ' athletic display during those games earned him the nickname . His other nickname is `` Neon Deion
'' . Following his first Super Bowl victory with the San Francisco 49ers , Sanders hosted Saturday Night Live , broadcast on February 18 , 1995 .
</item 9>

<item 10>
Deion Sanders
After getting into a scuffle with his former Falcon teammate Andre Rison , Sanders intercepted a pass from quarterback Jeff George and returned it 93 yards
while mockingly staring down the entire Falcons sideline before high - stepping into the end zone . Sanders was later voted the 1994 NFL Defensive Player of the
Year and recorded an interception in the 49ers 49 -- 26 victory over the San Diego Chargers in Super Bowl XXIX , earning him his first championship ring .
Dallas Cowboys ( edit ) Following his successful season with the 49ers , Sanders , along with his agent Eugene Parker , courted numerous teams in need of a
cornerback . The several teams in the `` Deion Sweepstakes '' , as it was called by the media , were the Philadelphia Eagles , Oakland Raiders , Miami Dolphins
, New Orleans Saints , San Francisco 49ers and the Dallas Cowboys , who had lost their starting cornerback Kevin Smith to injury for the rest of the season .
</item 10>
=== END ITEM id=2b63298f615363b1 ===

=== ITEM id=3eb958c7f8f1a855 ===
Gold sentences and facts:
Sentence 0: Manchester United Football Club is a professional football club based in Old Trafford , Greater Manchester , England , that competes in the Premier League , the top flight of English football . Nicknamed `` the Red Devils '' , the club was founded as Newton Heath LYR Football Club in 1878 , changed its name to Manchester United in 1902 and moved to its current stadium , Old Trafford , in 1910 .
  Fact 0.0: Manchester United Football Club is a professional football club.
  Fact 0.1: Manchester United Football Club is based in Old Trafford, Greater Manchester, England.
  Fact 0.2: Manchester United Football Club competes in the Premier League.
  Fact 0.3: The Premier League is the top flight of English football.
  Fact 0.4: Manchester United is nicknamed the Red Devils.
  Fact 0.5: The club was founded as Newton Heath LYR Football Club in 1878.
  Fact 0.6: The club changed its name to Manchester United in 1902.
  Fact 0.7: The club moved to its current stadium, Old Trafford, in 1910.

Retrieved items:
<item 1>
Manchester United F.C.
Some expansion took place in 1887 , and in 1891 , Newton Heath used its minimal financial reserves to purchase two grandstands , each able to hold 1,000
spectators . Although attendances were not recorded for many of the earliest matches at North Road , the highest documented attendance was approximately 15,000
for a First Division match against Sunderland on 4 March 1893 . A similar attendance was also recorded for a friendly match against Gorton Villa on 5 September
1889 . In June 1893 , after the club was evicted from North Road by its owners , Manchester Deans and Canons , who felt it was inappropriate for the club to
charge an entry fee to the ground , secretary A.H. Albut procured the use of the Bank Street ground in Clayton . It initially had no stands , by the start of
the 1893 -- 94 season , two had been built ; one spanning the full length of the pitch on one side and the other behind the goal at the `` Bradford end '' . At
the opposite end , the `` Clayton end '' , the ground had been `` built up , thousands thus being provided for '' .
</item 1>

<item 2>
Manchester United F.C.
Nicknamed `` the Red Devils '' , the club was founded as Newton Heath LYR Football Club in 1878 , changed its name to Manchester United in 1902 and moved to its
current stadium , Old Trafford , in 1910 . Manchester United have won a record 20 League titles , 12 FA Cups , 5 League Cups and a record 21 FA Community
Shields . The club has also won three UEFA Champions Leagues , one UEFA Europa League , one UEFA Cup Winners ' Cup , one UEFA Super Cup , one Intercontinental
Cup and one FIFA Club World Cup . In 1998 -- 99 , the club became the first in the history of English football to achieve the treble of the Premier League , the
FA Cup and the UEFA Champions League . In 2016 -- 17 , by winning the UEFA Europa League , they became one of five clubs to have won all three main UEFA club
competitions . In addition , they became the only professional English club to have won every ongoing honour available to the first team that is organised by a
national or international governing body . The 1958 Munich air disaster claimed the lives of eight players .
</item 2>

<item 3>
Manchester United F.C.
The record was pushed steadily upwards before reaching its peak on 31 March 2007 , when 76,098 spectators saw Manchester United beat Blackburn Rovers 4 -- 1 ,
with just 114 seats ( 0.15 per cent of the total capacity of 76,212 ) unoccupied . In 2009 , reorganisation of the seating resulted in a reduction of capacity
by 255 to 75,957 . Manchester United has the second highest average attendance of European football clubs only behind Borussia Dortmund . Support Manchester
United is one of the most popular football clubs in the world , with one of the highest average home attendance in Europe . The club states that its worldwide
fan base includes more than 200 officially recognised branches of the Manchester United Supporters Club ( MUSC ) , in at least 24 countries . The club takes
advantage of this support through its worldwide summer tours . Accountancy firm and sports industry consultants Deloitte estimate that Manchester United has 75
million fans worldwide , while other estimates put this figure closer to 333 million .
</item 3>

<item 4>
Manchester United F.C.
Non-League ( tiers 5 -- 11 ) 5 None 6 Curzon Ashton F.C. United of Manchester Salford City Stockport County 7 Altrincham Ashton United Stalybridge Celtic 8
Atherton Collieries Droylsden Hyde United Mossley Radcliffe Borough Ramsbottom United Trafford 9 Abbey Hey Ashton Athletic Irlam Maine Road West Didsbury &
Chorlton 10 Atherton Laburnum Rovers Chadderton Cheadle Town Daisy Hill Prestwich Heys Stockport Town 11 Ashton Town Eagley Wythenshawe Amateurs Competitions
Manchester Premier Cup Manchester Senior Cup Venues Boundary Park Bower Fold Broadhurst Park Butcher 's Arms Ground City of Manchester Stadium DW Stadium
Edgeley Park Ewen Fields Gigg Lane Macron Stadium Moss Lane Old Trafford Spotland Stadium Other topics Manchester derby Laureus World Sports Award for Team of
the Year winners 2000 : Manchester United F.C.
</item 4>

<item 5>
Manchester United F.C.
These were dismantled in 1987 and replaced by a lighting system embedded in the roof of each stand , which remains in use today . The Taylor Report 's
requirement for an all - seater stadium lowered capacity at Old Trafford to around 44,000 by 1993 . In 1995 , the North Stand was redeveloped into three tiers ,
restoring capacity to approximately 55,000 . At the end of the 1998 -- 99 season , second tiers were added to the East and West Stands , raising capacity to
around 67,000 , and between July 2005 and May 2006 , 8,000 more seats were added via second tiers in the north - west and north - east quadrants . Part of the
new seating was used for the first time on 26 March 2006 , when an attendance of 69,070 became a new Premier League record .
</item 5>

<item 6>
Manchester United F.C.
on BBC Sport : Club news -- Recent results and fixtures Manchester United at Sky Sports Manchester United at Premier League Manchester United Football Club
Players Managers Reserves & Academy Records & statistics Current season History 1878 -- 1945 1945 -- 1969 1969 -- 1986 1986 -- 2013 Seasons Europe Munich air
disaster Munich air disaster Busby Babes United Home stadium North Road ( 1878 -- 1893 ) Bank Street ( 1893 -- 1910 ) Old Trafford ( 1910 -- present ) Stretford
End Sir Matt Busby Way Training ground The Cliff Trafford Training Centre Players 100 + appearances 25 -- 99 appearances 1 -- 24 appearances Sir Matt Busby
Player of the Year Rivalries League record by opponent Manchester derby Liverpool rivalry Leeds United rivalry Arsenal rivalry Supporters Independent Manchester
United Supporters Association Manchester United Supporters ' Trust Red Army Media MUTV MU Radio Inside United Red Issue Red News Related articles Fergie 's
Fledglings United Trinity F.C. United of Manchester Glazer ownership Mascots 1915 British football betting scandal Railway station The Class of ' 92 Book :
Manchester United F.C. Category : Manchester United F.C. Portal
</item 6>

<item 7>
Manchester United F.C.
Manchester United F.C. - Wikipedia Manchester United F.C. Jump to : navigation , search `` MUFC '' and `` Manchester United '' redirect here . For other uses ,
see MUFC ( disambiguation ) . For the basketball team , see Manchester Eagles . Not to be confused with F.C. United of Manchester . Manchester United Full name
Manchester United Football Club Nickname ( s ) The Red Devils Short name MUFC Founded 1878 ; 140 years ago ( 1878 ) , as Newton Heath LYR F.C. 1902 ; 116 years
ago ( 1902 ) , as Manchester United F.C. Ground Old Trafford Ground Capacity 74,994 Owner Manchester United plc ( NYSE : MANU ) Co-chairmen Joel and Avram
Glazer Manager José Mourinho League Premier League 2016 -- 17 Premier League , 6th of 20 Website Club website Home colours Away colours Third colours Current
season Manchester United Football Club is a professional football club based in Old Trafford , Greater Manchester , England , that competes in the Premier
League , the top flight of English football .
</item 7>

<item 8>
Manchester United F.C.
Within four years , the stadium had cover on all four sides , as well as the ability to hold approximately 50,000 spectators , some of whom could watch from the
viewing gallery atop the Main Stand . However , following Manchester United 's first league title in 1908 and the FA Cup a year later , it was decided that Bank
Street was too restrictive for Davies ' ambition ; in February 1909 , six weeks before the club 's first FA Cup title , Old Trafford was named as the home of
Manchester United , following the purchase of land for around £ 60,000 . Architect Archibald Leitch was given a budget of £ 30,000 for construction ; original
plans called for seating capacity of 100,000 , though budget constraints forced a revision to 77,000 . The building was constructed by Messrs Brameld and Smith
of Manchester . The stadium 's record attendance was registered on 25 March 1939 , when an FA Cup semi-final between Wolverhampton Wanderers and Grimsby Town
drew 76,962 spectators .
</item 8>

<item 9>
Manchester United F.C.
Bombing in the Second World War destroyed much of the stadium ; the central tunnel in the South Stand was all that remained of that quarter . After the war ,
the club received compensation from the War Damage Commission in the amount of £ 22,278 . While reconstruction took place , the team played its `` home '' games
at Manchester City 's Maine Road ground ; Manchester United was charged £ 5,000 per year , plus a nominal percentage of gate receipts . Later improvements
included the addition of roofs , first to the Stretford End and then to the North and East Stands . The roofs were supported by pillars that obstructed many
fans ' views , and they were eventually replaced with a cantilevered structure . The Stretford End was the last stand to receive a cantilevered roof , completed
in time for the 1993 -- 94 season . First used on 25 March 1957 and costing £ 40,000 , four 180 - foot ( 55 m ) pylons were erected , each housing 54 individual
floodlights .
</item 9>

<item 10>
Manchester United F.C.
Newton Heath 's first league match at Bank Street was played against Burnley on 1 September 1893 , when 10,000 people saw Alf Farman score a hat - trick ,
Newton Heath 's only goals in a 3 -- 2 win . The remaining stands were completed for the following league game against Nottingham Forest three weeks later . In
October 1895 , before the visit of Manchester City , the club purchased a 2,000 - capacity stand from the Broughton Rangers rugby league club , and put up
another stand on the `` reserved side '' ( as distinct from the `` popular side '' ) . However , weather restricted the attendance for the Manchester City match
to just 12,000 . When the Bank Street ground was temporarily closed by bailiffs in 1902 , club captain Harry Stafford raised enough money to pay for the club 's
next away game at Bristol City and found a temporary ground at Harpurhey for the next reserves game against Padiham . Following financial investment , new club
president John Henry Davies paid £ 500 for the erection of a new 1,000 - seat stand at Bank Street .
</item 10>
=== END ITEM id=3eb958c7f8f1a855 ===

=== ITEM id=9eefc4a56107e8a8 ===
Gold sentences and facts:
Sentence 0: Teri Garr as Mrs. Ramsay , Alec 's mother
  Fact 0.0: Teri Garr plays Mrs. Ramsay.
  Fact 0.1: Mrs. Ramsay is Alec's mother.

Retrieved items:
<item 1>
The Black Stallion (film)
In 2002 , it was selected for preservation in the United States National Film Registry by the Library of Congress as being `` culturally , historically , or
aesthetically significant '' . Also , the film is recognized by American Film Institute in these lists : 2001 : AFI 's 100 Years ... 100 Thrills -- Nominated
2005 : AFI 's 100 Years of Film Scores -- Nominated 2006 : AFI 's 100 Years ... 100 Cheers -- # 64 Sequel , prequel and television series ( edit ) The film was
followed in 1983 by a sequel , The Black Stallion Returns , which also starred Reno . There was also a television series called The Adventures of the Black
Stallion which aired from 1990 to 1993 and starred Mickey Rooney and Richard Ian Cox . In 2003 , a 50 - minute prequel called The Young Black Stallion , was
shot and released for IMAX theaters . See also ( edit ) The Black Stallion books References ( edit ) Jump up ^ `` The Black Stallion , Box Office Information ''
. The Numbers .
</item 1>

<item 2>
The Black Stallion (film)
Alec 's father rescues Alec from his attacker but the life jacket is lost and father & son are separated . The black stallion breaks free from his compartment
and then jumps into the sea . Alec himself is swept overboard by a gigantic wave . Once in the water , he swims toward the stallion and managed to grab hold of
the ropes that were previously restraining the stallion . Alec wakes on the shore of a deserted island and starts to explore . He finds the stallion caught in
his restraints with the ropes stuck between the rocks . With his knife , Alec manages to free the stallion once again and the stallion runs away . For a time ,
the two keep their distance . Alec discovers means to survive by catching fish and seaweed . As Alec suddenly faces a cobra eye to eye , the Black comes to the
rescue and kills the snake , only to run off again . By now , Alec decides to try to get closer to the horse and offer him some seaweed . The hungry stallion
finds himself unable to resist , but visibly struggles with his distrust for humans .
</item 2>

<item 3>
The Black Stallion (film)
The Black stallion ( film ) - wikipedia The Black stallion ( film ) Jump to : navigation , search The Black Stallion 1979 poster Directed by Carroll Ballard
Produced by Fred Roos Francis Ford Coppola Tom Sternberg Screenplay by Melissa Mathison Jeanne Rosenberg William D. Wittliff Based on The Black Stallion by
Walter Farley Starring Kelly Reno Mickey Rooney Teri Garr Music by Carmine Coppola Cinematography Caleb Deschanel Edited by Robert Dalva Production company
American Zoetrope Distributed by United Artists Release date October 17 , 1979 ( 1979 - 10 - 17 ) Running time 118 minutes Country United States Language
English Budget $2.7 million Box office $37.8 million The Black Stallion is a 1979 American adventure film based on the 1941 classic children 's novel The Black
Stallion by Walter Farley . But the film starts in 1946 , five years after the book was published . It tells the story of Alec Ramsey , who is shipwrecked on a
deserted island with a wild Arabian stallion whom he befriends .
</item 3>

<item 4>
The Black Stallion (film)
American Film Institute . Retrieved 2016 - 08 - 20 . External links ( edit ) Official website The Black Stallion on IMDb The Black Stallion at AllMovie The
Black Stallion at Rotten Tomatoes The Black Stallion at Box Office Mojo The Black Stallion at the TCM Movie Database The Black Stallion by Walter Farley Books
The Black Stallion book series Films The Black Stallion ( 1979 ) The Black Stallion Returns ( 1983 ) The Young Black Stallion ( 2003 ) Television The Adventures
of the Black Stallion ( 1990 - 1993 ) Related Cass Ole El Mokhtar Docs Keepin Time Films directed by Carroll Ballard The Black Stallion ( 1979 ) Never Cry Wolf
( 1983 ) Nutcracker : The Motion Picture ( 1986 ) Wind ( 1992 ) Fly Away Home ( 1996 ) Duma ( 2005 ) Retrieved from `` https://en.wikipedia.org/w/index.
</item 4>

<item 5>
The Black Stallion (film)
1 John Buchanan as Jockey No. 2 Kristen Vigard as Becky Fausto Tozzi as Rescue Captain John Karlsen as Archeologist Leopoldo Trieste as Priest Frank Cousins as
African Chieftain Don Hudson as Zaurog Marne Maitland as Drake Captain Tom Dahlgren as Veterinarian Horses ( edit ) Cass Ole , a champion Arabian stallion , was
featured in most of the movie 's scenes , with Fae Jur , another black Arabian stallion , being his main double . Fae Jur 's main scene is the one where Alec is
trying to gain the trust of the Black on the beach . Two other stunt doubles were used for running , fighting and swimming scenes . El Mokhtar , an Egyptian
Arabian racehorse , was the producers ' first choice to portray the Black , but they were unable to secure his services for the film from his owners , who
declined any offers . He does appear in The Black Stallion Returns , alongside Cass Ole , by which time the studio bought out the syndicate of owners in order
to secure El Mokhtar 's services .
</item 5>

<item 6>
The Black Stallion (film)
He falls behind , but the Black wo n't stop . When Alec regains his balance , the Black is well on his way to catch up with his opponents . Alec now encourages
the Black to run as fast as he can , remembering the wild rides on the island , as they catch up . The Black eventually wins by two lengths . Cast ( edit )
Kelly Reno as Alec Ramsay Mickey Rooney as Henry Dailey Teri Garr as Mrs. Ramsay , Alec 's mother Hoyt Axton as Mr. Ramsay , Alec 's father Clarence Muse as
Snoe Michael Higgins as Jim Neville Ed McNamara as Jake Doghmi Larbi as Arab John Burton as Jockey No. 1 John Buchanan as Jockey No.
</item 6>

<item 7>
The Black Stallion (film)
After being rescued , they are set on entering a race challenging two champion horses . The film is adapted by Melissa Mathison , Jeanne Rosenberg and William
D. Wittliff . It is directed by Carroll Ballard . The movie stars Kelly Reno , Mickey Rooney , Teri Garr , Hoyt Axton , and the Arabian horse Cass Ole . The
film features music by Carmine Coppola , the father of Hollywood producer Francis Ford Coppola , who was the executive producer of the film . Contents ( hide )
1 Plot 2 Cast 2.1 Horses 3 Awards and honors 3.1 Academy Awards 3.2 Golden Globe Awards 3.3 British Academy Awards 3.4 LA Film Critics Awards 3.5 Others 4
Sequel , prequel and television series 5 See also 6 References 7 External links Plot ( edit ) In 1946 Alec Ramsey is aboard the steamer Drake off the coast of
North Africa , where he sees a wild black stallion being forced into a makeshift stable compartment and heavily restrained by ropes leading to his halter .
</item 7>

<item 8>
The Black Stallion (film)
wikipedia.org/w/index.php?title=The_Black_Stallion_(film)&oldid=836268326 '' Categories : 1979 films English - language films American children 's films Films
about horses United States National Film Registry films United Artists films American Zoetrope films Directorial debut films Films shot in Sardinia Films
directed by Carroll Ballard Films that won the Best Sound Editing Academy Award Horse racing films Screenplays by Melissa Mathison American films Films set on
islands Films set in 1946 Hidden categories : Pages using div col with deprecated parameters Talk Contents About Wikipedia Català Dansk Deutsch Español فارسی
Français Bahasa Indonesia Italiano Bahasa Melayu Nederlands 日本 語 Polski Português Русский Svenska 6 more Edit links This page was last edited on 13 April 2018 ,
at 18 : 25 . About Wikipedia
</item 8>

<item 9>
The Black Stallion (film)
Henry allows Alec to keep the Black in his barn . When Alec wonders how fast the Black is , Alec and Henry decide to train the Black for the racetrack , while
Henry teaches Alec how to be a jockey . The Black surprises Henry with his speed . Henry immediately starts plotting to get the Black into a match race between
two of the country 's current champions , to be held at Santa Anita , California . To do that , he sets up a secret demonstration at night where the press can
witness his speed , keeping the identity of Alec and the Black secret . The news about the mystery horse is soon widespread and the Black is entered into the
race . The race is the most anticipated horse - racing event of the year . Before the two champions and the Black enter the starting gate , the Black gets into
a fight with one of his opponents , wounding his leg . Alec does not see the wound until he is in the gate . As he dismounts , the bell rings and the horses
take off . Alec desperately tries to stay on his horse and trying to stop him . He falls behind , but the Black wo n't stop .
</item 9>

<item 10>
The Black Stallion (film)
The hungry stallion finds himself unable to resist , but visibly struggles with his distrust for humans . Eventually , the hunger wins and he takes Alec 's
offer ; their bond has been sealed and the two are now inseparable . Alec even manages to ride the unbroken horse , after many times falling off the horse . One
day , a fishing ship arrives , rescuing both Alec and the stallion . Back home in Flushing , NY , Alec is given a hero 's welcome . The Black has a temporary
home in Alec 's backyard , but a garbage man not knowing that there is a wild horse in the backyard is chased by the Black , who races off down the street after
being spooked by a passing car . Alec chases after him through every part of town , but loses track of him . The next day , Alec meets Snoe ( and Napoleon ) who
tell him where the Black is . Alec finds the stallion in the barn of Henry Dailey , a retired jockey and racehorse trainer , who apparently spent all night
catching the Black . Henry allows Alec to keep the Black in his barn .
</item 10>
=== END ITEM id=9eefc4a56107e8a8 ===

=== ITEM id=293ce03bb5cc2247 ===
Gold sentences and facts:
Sentence 0: The Proud Family is a soundtrack album for the show of the same name . A combination of both original tunes by the characters in the show , popular hits by popular music artists such as Alicia Keys , Jhené Aiko , India Arie , and Solange Knowles & Destiny 's Child ( who perform the theme song ) , and classic soul music from artists such as Aretha Franklin and The O'Jays .
  Fact 0.0: The Proud Family is a soundtrack album for the show of the same name.
  Fact 0.1: The album combines original tunes by the characters in the show.
  Fact 0.2: The album includes popular hits by popular music artists such as Alicia Keys, Jhené Aiko, India Arie, and Solange Knowles and Destiny's Child.
  Fact 0.3: Destiny's Child performs the theme song.
  Fact 0.4: The album includes classic soul music from artists such as Aretha Franklin and The O'Jays.

Retrieved items:
<item 1>
The Proud Family (soundtrack)
The Proud Family ( soundtrack ) - Wikipedia The Proud Family ( soundtrack ) Jump to : navigation , search This article does not cite any sources . Please help
improve this article by adding citations to reliable sources . Unsourced material may be challenged and removed . ( December 2009 ) ( Learn how and when to
remove this template message ) The Proud Family Soundtrack album by Various Artists Released April 20 , 2004 August 16 , 2005 ( Re-release ) Genre Soul , hip
hop , hip hop soul Label Walt Disney The Proud Family is a soundtrack album for the show of the same name . A combination of both original tunes by the
characters in the show , popular hits by popular music artists such as Alicia Keys , Jhené Aiko , India Arie , and Solange Knowles & Destiny 's Child ( who
perform the theme song ) , and classic soul music from artists such as Aretha Franklin and The O'Jays . References ( edit ) This soundtrack - related article is
a stub . You can help Wikipedia by expanding it .
</item 1>

<item 2>
The Proud Family (soundtrack)
You can help Wikipedia by expanding it . Retrieved from `` https://en.wikipedia.org/w/index.php?title=The_Proud_Family_(soundtrack)&oldid=638462881 ''
Categories : Television animation soundtracks 2004 soundtracks Hip hop soundtracks Soul soundtracks Walt Disney Records soundtracks Soundtrack stubs Hidden
categories : Articles lacking sources from December 2009 All articles lacking sources Articles with hAudio microformats Music infoboxes with deprecated
parameters All stub articles Talk About Wikipedia Add links This page was last edited on 17 December 2014 , at 07 : 09 . Text is available under the Creative
Commons Attribution - ShareAlike License ; additional terms may apply . By using this site , you agree to the Terms of Use and Privacy Policy . Wikipedia ® is a
registered trademark of the Wikimedia Foundation , Inc. , a non-profit organization . About Wikipedia
</item 2>
=== END ITEM id=293ce03bb5cc2247 ===

=== ITEM id=2df765cb83ade8d7 ===
Gold sentences and facts:
Sentence 0: `` Ca n't Get You Out of My Head '' is a song recorded by Australian singer Kylie Minogue for her eighth studio album , titled Fever , which she released in 2001 . The song was released in Australia by Parlophone as the lead single from the album on 8 September 2001 . It was released on 17 September 2001 in the United Kingdom . In the United States , the single was released on 18 February 2002 . Jointly written , composed , and produced by Cathy Dennis and Rob Davis , `` Ca n't Get You Out of My Head '' is a midtempo dance - pop song which lyrically details its narrator 's obsession towards her lover . The song is famous for its `` la la la '' hook .
  Fact 0.0: Can't Get You Out of My Head is a song recorded by Australian singer Kylie Minogue.
  Fact 0.1: Can't Get You Out of My Head was recorded for Kylie Minogue's eighth studio album, Fever.
  Fact 0.2: Fever was released by Kylie Minogue in 2001.
  Fact 0.3: The song was released in Australia by Parlophone as the lead single from the album Fever on 8 September 2001.
  Fact 0.4: The song was released in the United Kingdom on 17 September 2001.
  Fact 0.5: The single was released in the United States on 18 February 2002.
  Fact 0.6: Can't Get You Out of My Head was jointly written, composed, and produced by Cathy Dennis and Rob Davis.
  Fact 0.7: Can't Get You Out of My Head is a midtempo dance-pop song.
  Fact 0.8: The song lyrically details its narrator's obsession with her lover.
  Fact 0.9: The song is famous for its "la la la" hook.

Retrieved items:
<item 1>
Can't Get You Out of My Head
By using this site , you agree to the Terms of Use and Privacy Policy . Wikipedia ® is a registered trademark of the Wikimedia Foundation , Inc. , a non-profit
organization . About Wikipedia
</item 1>

<item 2>
Can't Get You Out of My Head
The song was recorded using Cubase music software , which Davis ran on his Mac computer . Davis began playing an acoustic guitar and ran a 125 beats per minute
drum loop , on which Dennis , herself a singer who had enjoyed chart success in the United States , began singing the line `` I just ca n't get you out of my
head '' in the key of D minor . After three and a half hours , the demo was recorded and the vocals were laid afterwards . Dennis called their recording setup
for the song `` the most primitive set - up you could imagine ! Different producers work in different ways . But it 's good to be reminded you do n't have to be
reliant on equipment . A song is about melody and lyrics and being able to take something away in your memory that is going to haunt you . '' She also regarded
their production as a `` very natural and fluid process , '' saying : `` We know how hard we work sometimes to write songs and then spend months picking them to
pieces , but this was the easiest process , the chemicals were all happy and working together . ''
</item 2>

<item 3>
Can't Get You Out of My Head
Through the lyrics of the song , its singer expresses an obsession with an anonymous figure . Dorian Lynskey from The Guardian termed the song a `` mystery ''
as Minogue never reveals the identity of her object of infatuation . The critic suggested that the person Minogue is referring to is either `` a partner , an
evasive one - night stand or someone who does n't know she exists . '' Writing for the same newspaper , Everett True identified a `` darker element '' in the
simple lyrics and felt this sentiment was echoed in Minogue 's restrained vocals . Further , True emphasised that while Minogue 's 1987 single `` I Should Be So
Lucky '' had presented an optimistic romantic future , `` Ca n't Get You Out of My Head '' focuses on an `` unhealthy '' and potentially destructive obsession .
He also noted that in the former song , Minogue played `` the wide - eyed ingénue with alacrity , '' but in the latter song she is aware of the harmful nature
of her infatuation , calling it a `` desire that is wholly dependent on her own self - control . ''
</item 3>

<item 4>
Can't Get You Out of My Head
The music , excluding the guitar part , was programmed using Korg Triton workstation via MIDI . Dennis later remarked : `` Even though Kylie was n't the first
artist to be offered the song , I do n't believe it was meant to go to anyone other than Kylie , and I do n't believe anyone else would have done the incredible
job she did with it , with the video , looking super-hot ! '' Composition ( edit ) `` Ca n't Get You Out of My Head '' A 21 - second sample of `` Ca n't Get You
Out of My Head '' containing the hook and part of the chorus of the song . The song is known for its `` la la la '' hook . Problems playing this file ? See
media help . `` Ca n't Get You Out of My Head '' is a `` robotic '' midtempo dance - pop song , with a tempo of 126 beats per minute .
</item 4>

<item 5>
Can't Get You Out of My Head
My A&R at the time , Miles Leonard and Jamie Nelson , said , `` We 've got something . Come into the office . We 've got to play you something . '' So I went to
the EMI office , and I had the same reaction you did , about 20 seconds in . I could n't even fathom what I was hearing . It just ... did something . I was
beside myself . Then at the end of the song , panic set in . I was saying `` Are you sure we 've got this song ? Do n't tell me that we do n't ! Is it secured ?
Can we have it ? '' And we did ! And that kick - started a whole different phase in my career . -- Minogue , about her reaction to hearing the demo of the song
. `` Ca n't Get You Out of My Head '' was jointly written , composed , and produced by Cathy Dennis and Rob Davis . Dennis and Davis had been brought together
by British artist manager Simon Fuller , who wanted the duo to come up with a song for British pop group S Club 7 .
</item 5>

<item 6>
Can't Get You Out of My Head
It was released on 17 September 2001 in the United Kingdom . In the United States , the single was released on 18 February 2002 . Jointly written , composed ,
and produced by Cathy Dennis and Rob Davis , `` Ca n't Get You Out of My Head '' is a midtempo dance - pop song which lyrically details its narrator 's
obsession towards her lover . The song is famous for its `` la la la '' hook . In addition to positive reviews from music critics , `` Ca n't Get You Out of My
Head '' found commercial success on a large scale . It peaked at number one on the charts of Austria , Belgium , France , Germany , Italy , Poland , Switzerland
, the United Kingdom , and every other European country excluding Finland . It also topped the charts of Minogue 's native country Australia , Canada and New
Zealand . In the United States , the song peaked at number seven on the US Billboard Hot 100 chart , becoming Minogue 's biggest hit in the region since `` The
Loco - Motion '' . `` Ca n't Get You Out of My Head '' reportedly reached number one in 40 countries across the globe .
</item 6>

<item 7>
Can't Get You Out of My Head
Ca n't Get You Out of My Head - Wikipedia Ca n't Get You Out of My Head Jump to : navigation , search `` Ca n't Get You Out of My Head '' Single cover used for
CD # 02 and digital releases of the song .
</item 7>

<item 8>
Can't Get You Out of My Head
: 26 Digital download `` Ca n't Get You Out of My Head '' -- 3 : 52 `` Boy '' -- 3 : 48 Charts ( edit ) Weekly charts ( edit ) Chart ( 2001 -- 02 ) Peak
position Australia ( ARIA ) Austria ( Ö3 Austria Top 40 ) Belgium ( Ultratop 50 Flanders ) Belgium ( Ultratop 50 Wallonia ) Brazil ( ABPD ) Canada ( Canadian
Hot 100 ) Denmark ( Tracklisten ) Finland ( Suomen virallinen lista ) 5 France ( SNEP ) Germany ( Official German Charts ) Greece ( IFPI Greece ) Ireland ( IRMA
) Italy ( FIMI ) Netherlands ( Single Top 100 ) New Zealand ( Recorded Music NZ ) Norway ( VG - lista ) Poland ( Polish Singles Chart ) Portugal ( Billboard )
Romania ( Romanian Top 100 ) Scotland ( Official Charts Company ) South Africa ( RiSA ) Spain ( PROMUSICAE ) Sweden ( Sverigetopplistan ) Switzerland (
Schweizer Hitparade ) UK Singles ( Official Charts Company ) US Billboard Hot 100 7 US Adult Top 40 ( Billboard ) 23 US Dance Club Songs ( Billboard )
</item 8>

<item 9>
Can't Get You Out of My Head
In 2012 , Priya Elan from NME ranked the song at number four on his `` The Greatest Pop Songs in History '' list , saying : `` It was unlike any song I remember
hearing before . '' In 2012 , The Guardian included the song in their list of `` The Best Number One Records , '' labelling it to be `` sleek , Arctic - blue
minimalism , like an emotionally thwarted retelling of Donna Summer 's `` I Feel Love . '' In the same year , PRS for Music , a UK copyright collection society
and performance rights organisation which collects royalties on behalf of songwriters and composers , named `` Ca n't Get You Out of My Head '' the `` Most
Popular Song of the Decade '' as it received the highest airplay and live covers in the 2000s ( decade ) . Minogue performing `` Ca n't Get You Out of My Head
'' during her KylieX Tour ( 2008 ) .
</item 9>

<item 10>
Can't Get You Out of My Head
In 2013 , a survey of 700 people was conducted as part of the Manchester Science Festival to find out which song they considered the `` catchiest , '' and `` Ca
n't Get You Out of My Head '' topped the poll . Lee Barron , in his essay The Seven Ages of Kylie Minogue : Postmodernism , Identity , and Performative Mimicry
, noted that the song `` further established Minogue 's cultural and commercial relevance in the new millennium . '' He remarked that the song `` with its
hypnotic ' la la la ' refrain and the deceptively uncomplicated , catchily repetitive beats and synth - sound , marked yet another clearly defined image
transformation from the camp - infused Light Years to an emphasis upon a cool , machine - like sexuality , a trait clearly identifiable within the promotional
video for ' Ca n't Get You Out of My Head . '
</item 10>
=== END ITEM id=2df765cb83ade8d7 ===

=== ITEM id=dd817835119b71f1 ===
Gold sentences and facts:
Sentence 0: The story of Shakespeare 's Hamlet was derived from the legend of Amleth , preserved by 13th - century chronicler Saxo Grammaticus in his Gesta Danorum , as subsequently retold by the 16th - century scholar François de Belleforest . Shakespeare may also have drawn on an earlier Elizabethan play known today as the Ur - Hamlet , though some scholars believe he himself wrote the Ur - Hamlet , later revising it to create the version of Hamlet we now have . He almost certainly wrote his version of the title role for his fellow actor , Richard Burbage , the leading tragedian of Shakespeare 's time . In the 400 years since its inception , the role has been performed by numerous highly acclaimed actors in each successive century .
  Fact 0.0: The story of Shakespeare's Hamlet was derived from the legend of Amleth.
  Fact 0.1: The legend of Amleth was preserved by 13th-century chronicler Saxo Grammaticus in his Gesta Danorum.
  Fact 0.2: The legend of Amleth was subsequently retold by 16th-century scholar François de Belleforest.
  Fact 0.3: Shakespeare may also have drawn on an earlier Elizabethan play known today as the Ur-Hamlet.
  Fact 0.4: Some scholars believe Shakespeare himself wrote the Ur-Hamlet.
  Fact 0.5: Some scholars believe Shakespeare later revised the Ur-Hamlet to create the version of Hamlet we now have.
  Fact 0.6: Shakespeare almost certainly wrote his version of the title role for his fellow actor, Richard Burbage.
  Fact 0.7: Richard Burbage was the leading tragedian of Shakespeare's time.
  Fact 0.8: In the 400 years since Hamlet's inception, the role of Hamlet has been performed by numerous highly acclaimed actors in each successive century.

Retrieved items:
<item 1>
Hamlet
( father ) Mary Arden ( mother ) Gilbert Shakespeare ( brother ) Joan Shakespeare ( sister ) Edmund Shakespeare ( brother ) Richard Shakespeare ( grandfather )
John Hall ( son - in - law ) Thomas Quiney ( son - in - law ) Thomas Nash ( grandson - in - law ) * Shakespeare and other authors Lost William Shakespeare 's
Hamlet Characters Hamlet Claudius Gertrude Ghost Polonius Laertes Ophelia Horatio Rosencrantz and Guildenstern Fortinbras The Gravediggers Yorick Soliloquies ``
To be , or not to be '' `` Mortal coil '' `` What a piece of work is a man '' `` Speak the speech '' Words and phrases `` The lady doth protest too much ,
methinks '' `` Thy name is '' `` Primrose path '' Terminology Dumbshow Induction Quiddity Substitution Sources Criticism Legend of Hamlet The Spanish Tragedy Ur
- Hamlet Critical approaches Bibliographies Horwendill Saxo Grammaticus House of Gonzaga Damon and Pythias Influence Common phrases from Hamlet Hamlet in
popular culture References to Ophelia Language of flowers Human skull symbolism Performances Moscow Art Theatre ( 1911 -- 1912 )
</item 1>

<item 2>
Hamlet
Jr. , in the American Company 's production in Philadelphia in 1759 . David Garrick 's iconic hand gesture expresses Hamlet 's shock at the first sight of the
ghost ( artist : unknown ) . John Philip Kemble made his Drury Lane debut as Hamlet in 1783 . His performance was said to be 20 minutes longer than anyone else
's , and his lengthy pauses provoked the suggestion by Richard Brinsley Sheridan that `` music should be played between the words '' . Sarah Siddons was the
first actress known to play Hamlet ; many women have since played him as a breeches role , to great acclaim . In 1748 , Alexander Sumarokov wrote a Russian
adaptation that focused on Prince Hamlet as the embodiment of an opposition to Claudius 's tyranny -- a treatment that would recur in Eastern European versions
into the 20th century . In the years following America 's independence , Thomas Apthorpe Cooper , the young nation 's leading tragedian , performed Hamlet among
other plays at the Chestnut Street Theatre in Philadelphia , and at the Park Theatre in New York .
</item 2>

<item 3>
Hamlet
Firm evidence for specific early performances of the play is scant . What is known is that the crew of the ship Red Dragon , anchored off Sierra Leone ,
performed Hamlet in September 1607 ; that the play toured in Germany within five years of Shakespeare 's death ; and that it was performed before James I in
1619 and Charles I in 1637 . Oxford editor George Hibbard argues that , since the contemporary literature contains many allusions and references to Hamlet (
only Falstaff is mentioned more , from Shakespeare ) , the play was surely performed with a frequency that the historical record misses . All theatres were
closed down by the Puritan government during the Interregnum . Even during this time , however , playlets known as drolls were often performed illegally ,
including one called The Grave - Makers based on Act 5 , Scene 1 of Hamlet . Restoration and 18th century ( edit ) Title page and frontispiece for Hamlet ,
Prince of Denmark : A Tragedy . As it is now acted at the Theatres - Royal in Drury - Lane and Covent - Garden . London , 1776 The play was revived early in the
Restoration .
</item 3>

<item 4>
Hamlet
London , 1776 The play was revived early in the Restoration . When the existing stock of pre-civil war plays was divided between the two newly created patent
theatre companies , Hamlet was the only Shakespearean favourite that Sir William Davenant 's Duke 's Company secured . It became the first of Shakespeare 's
plays to be presented with movable flats painted with generic scenery behind the proscenium arch of Lincoln 's Inn Fields Theatre . This new stage convention
highlighted the frequency with which Shakespeare shifts dramatic location , encouraging the recurrent criticism of his failure to maintain unity of place . In
the title role , Davenant cast Thomas Betterton , who continued to play the Dane until he was 74 . David Garrick at Drury Lane produced a version that adapted
Shakespeare heavily ; he declared : `` I had sworn I would not leave the stage till I had rescued that noble play from all the rubbish of the fifth act . I have
brought it forth without the grave - digger 's trick , Osrick , & the fencing match '' . The first actor known to have played Hamlet in North America is Lewis
Hallam . Jr.
</item 4>

<item 5>
Hamlet
Hamlet - wikipedia Hamlet Jump to : navigation , search This article is about the play by William Shakespeare . For other uses , see Hamlet ( disambiguation ) .
The American actor Edwin Booth as Hamlet , ca . 1870 The Tragedy of Hamlet , Prince of Denmark , often shortened to Hamlet ( / ˈhæmlɪt / ) , is a tragedy
written by William Shakespeare at an uncertain date between 1599 and 1602 . Set in Denmark , the play dramatises the revenge Prince Hamlet is called to wreak
upon his uncle , Claudius , by the ghost of Hamlet 's father , King Hamlet . Claudius had murdered his own brother and seized the throne , also marrying his
deceased brother 's widow . Hamlet is Shakespeare 's longest play , and is considered among the most powerful and influential works of world literature , with a
story capable of `` seemingly endless retelling and adaptation by others '' .
</item 5>

<item 6>
Hamlet
It was probably one of Shakespeare 's most popular works during his lifetime , and still ranks among his most performed , topping the performance list of the
Royal Shakespeare Company and its predecessors in Stratford - upon - Avon since 1879 . It has inspired many other writers -- from Johann Wolfgang von Goethe and
Charles Dickens to James Joyce and Iris Murdoch -- and has been described as `` the world 's most filmed story after Cinderella '' . The story of Shakespeare 's
Hamlet was derived from the legend of Amleth , preserved by 13th - century chronicler Saxo Grammaticus in his Gesta Danorum , as subsequently retold by the 16th
- century scholar François de Belleforest . Shakespeare may also have drawn on an earlier Elizabethan play known today as the Ur - Hamlet , though some scholars
believe he himself wrote the Ur - Hamlet , later revising it to create the version of Hamlet we now have . He almost certainly wrote his version of the title
role for his fellow actor , Richard Burbage , the leading tragedian of Shakespeare 's time .
</item 6>

<item 7>
Hamlet
In stark contrast to earlier opulence , William Poel 's 1881 production of the Q1 text was an early attempt at reconstructing the Elizabethan theatre 's
austerity ; his only backdrop was a set of red curtains . Sarah Bernhardt played the prince in her popular 1899 London production . In contrast to the ``
effeminate '' view of the central character that usually accompanied a female casting , she described her character as `` manly and resolute , but nonetheless
thoughtful ... ( he ) thinks before he acts , a trait indicative of great strength and great spiritual power '' . In France , Charles Kemble initiated an
enthusiasm for Shakespeare ; and leading members of the Romantic movement such as Victor Hugo and Alexandre Dumas saw his 1827 Paris performance of Hamlet ,
particularly admiring the madness of Harriet Smithson 's Ophelia . In Germany , Hamlet had become so assimilated by the mid-19th century that Ferdinand
Freiligrath declared that `` Germany is Hamlet '' . From the 1850s , the Parsi theatre tradition in India transformed Hamlet into folk performances , with
dozens of songs added .
</item 7>

<item 8>
Hamlet
Jenkins suggests that any personal satire may be found in the name `` Polonius '' , which might point to a Polish or Polonian connection . G.R. Hibbard
hypothesised that differences in names ( Corambis / Polonius : Montano / Raynoldo ) between the First Quarto and other editions might reflect a desire not to
offend scholars at Oxford University . Date ( edit ) John Barrymore as Hamlet ( 1922 ) `` Any dating of Hamlet must be tentative '' , cautions the New Cambridge
editor , Phillip Edwards . The earliest date estimate relies on Hamlet 's frequent allusions to Shakespeare 's Julius Caesar , itself dated to mid-1599 . The
latest date estimate is based on an entry , of 26 July 1602 , in the Register of the Stationers ' Company , indicating that Hamlet was `` latelie Acted by the
Lo : Chamberleyne his servantes '' . In 1598 , Francis Meres published his Palladis Tamia , a survey of English literature from Chaucer to its present day ,
within which twelve of Shakespeare 's plays are named .
</item 8>

<item 9>
Hamlet
He is dethroned by the spectre of an actor , and we shall never be able to keep the usurper out of our dreams . Maurice Maeterlinck in La Jeune Belgique ( 1890
) . Shakespeare 's day to the Interregnum ( edit ) Shakespeare almost certainly wrote the role of Hamlet for Richard Burbage . He was the chief tragedian of the
Lord Chamberlain 's Men , with a capacious memory for lines and a wide emotional range . Judging by the number of reprints , Hamlet appears to have been
Shakespeare 's fourth most popular play during his lifetime -- only Henry IV Part 1 , Richard III and Pericles eclipsed it . Shakespeare provides no clear
indication of when his play is set ; however , as Elizabethan actors performed at the Globe in contemporary dress on minimal sets , this would not have affected
the staging . Firm evidence for specific early performances of the play is scant .
</item 9>

<item 10>
Hamlet
In the 400 years since its inception , the role has been performed by numerous highly acclaimed actors in each successive century . Three different early
versions of the play are extant : the First Quarto ( Q1 , 1603 ) ; the Second Quarto ( Q2 , 1604 ) ; and the First Folio ( F1 , 1623 ) . Each version includes
lines and entire scenes missing from the others . The play 's structure and depth of characterisation have inspired much critical scrutiny . One such example is
the centuries - old debate about Hamlet 's hesitation to kill his uncle , which some see as merely a plot device to prolong the action , but which others argue
is a dramatisation of the complex philosophical and ethical issues that surround cold - blooded murder , calculated revenge , and thwarted desire . More
recently , psychoanalytic critics have examined Hamlet 's unconscious desires , while feminist critics have re-evaluated and attempted to rehabilitate the often
maligned characters of Ophelia and Gertrude . Contents ( hide ) 1 Characters 2 Plot 2.1 Act I 2.2 Act II 2.3 Act III 2.
</item 10>
=== END ITEM id=dd817835119b71f1 ===

=== ITEM id=492fe820dc8dc388 ===
Gold sentences and facts:
Sentence 0: The poem survives in a single manuscript , the Cotton Nero A.x. , which also includes three religious narrative poems : Pearl , Purity and Patience . All are thought to have been written by the same unknown author , dubbed the `` Pearl Poet '' or `` Gawain Poet '' , since all four are written in a North West Midland dialect of Middle English .
  Fact 0.0: The poem survives in a single manuscript, the Cotton Nero A.x.
  Fact 0.1: The Cotton Nero A.x. manuscript also includes three religious narrative poems: Pearl, Purity, and Patience.
  Fact 0.2: All four works are thought to have been written by the same unknown author.
  Fact 0.3: The unknown author is dubbed the "Pearl Poet" or "Gawain Poet".
  Fact 0.4: All four works are written in a North West Midland dialect of Middle English.

Retrieved items:
<item 1>
Sir Gawain and the Green Knight
6 Homoerotic interpretations 8 Modern adaptations 8.1 Books 8.2 Film and television 8.3 Theatre 8.4 Opera 9 References 10 External links Synopsis ( edit ) Sir
Gawain and the Green Knight ( from original manuscript , artist unknown ) In Camelot on New Year 's Day , King Arthur 's court is exchanging gifts and waiting
for the feasting to start when the king asks first to see or hear of an exciting adventure . A gigantic figure , entirely green in appearance and riding a green
horse , rides unexpectedly into the hall . He wears no armour but bears an axe in one hand and a holly bough in the other . Refusing to fight anyone there on
the grounds that they are all too weak to take him on , he insists he has come for a friendly `` Christmas game '' : someone is to strike him once with his axe
on condition that the Green Knight may return the blow in a year and a day . The splendid axe will belong to whoever takes him on .
</item 1>

<item 2>
Sir Gawain and the Green Knight
Sir Gawain and the Green Knight - wikipedia Sir Gawain and the Green Knight Jump to : navigation , search For the 1973 film adaptation , see Gawain and the
Green Knight ( film ) . Sir Gawain and the Green Knight First page of survived manuscript , circa 14th century . Author Anonymous Country Kingdom of England
Language Middle English Genre Poem , chivalric romance , Arthurian and alliterative verse . Publication date 14th century Sir Gawain and the Green Knight (
Middle English : Sir Gawayn and þe Grene Knyȝt ) is a late 14th - century Middle English chivalric romance . It is one of the best known Arthurian stories ,
with its plot combining two types of folklore motifs , the beheading game and the exchange of winnings . The Green Knight is interpreted by some as a
representation of the Green Man of folklore and by others as an allusion to Christ . Written in stanzas of alliterative verse , each of which ends in a rhyming
bob and wheel , it draws on Welsh , Irish , and English stories , as well as the French chivalric tradition .
</item 2>

<item 3>
Sir Gawain and the Green Knight
722 . ^ Jump up to : Hulbert , J.R. `` Syr Gawayn and the Grene Knyzt ( Concluded ) '' . Modern Philology. ( April 1916 ) 13.12 pp. 689 -- 730 . Jump up ^
Jackson , I. `` Sir Gawain 's Coat of Arms . '' The Modern Language Review . ( January 1920 ) 15.1 pp. 77 -- 79 . Jump up ^ Arthur , pp. 33 -- 34 . ^ Jump up to
: Arthur , p. 34 . Jump up ^ Arthur , p. 35 . Jump up ^ Cooke , Jessica ( 1998 ) . `` The Lady 's ' Blushing ' Ring in Sir Gawain and the Green Knight '' . The
Review of English Studies . 49 ( 193 ) : 1 -- 8 . doi : 10.1093 / res / 49.193. 1 . Jump up ^ Andrew , Malcolm ; Ronald Waldron ( 1996 ) . The Poems of the
Pearl Manuscript .
</item 3>

<item 4>
Sir Gawain and the Green Knight
`` King Arthur and King Cornwall '' . In Sir Gawain : Eleven Romances and Tales . Western Michigan University Medieval Institute Publications , 2000 . p. 427 .
ISBN 1 - 879288 - 59 - 1 ^ Jump up to : Lasater , Alice E. Spain to England : A Comparative Study of Arabic , European , and English Literature of the Middle
Ages . University Press of Mississippi , 1974 . ISBN 0 - 87805 - 056 - 6 Jump up ^ Rix , Michael M. `` A Re-Examination of the Castleton Garlanding '' .
Folklore . ( June 1953 ) 64.2 pp. 342 -- 344 Jump up ^ Buchanan , Alice ( June 1932 ) . `` The Irish Framework of Sir Gawain and the Green Knight '' . PLMA. 47
( 2 ) : 315 -- 338 . doi : 10.2307 / 457878 . Jump up ^ . Friedman , Albert B. Morgan le Fay in Sir Gawain and the Green Knight . Medieval Academy of America .
</item 4>

<item 5>
Sir Gawain and the Green Knight
Retrieved 2016 - 03 - 01 . External links ( edit ) Find more aboutSir Gawain and the Green Knightat Wikipedia 's sister projects Definitions from Wiktionary
Media from Wikimedia Commons News from Wikinews Quotations from Wikiquote Texts from Wikisource Learning resources from Wikiversity Online texts High -
resolution , full - sized scan of entire manuscript Sir Gawain and the Green Knight at Archive.org JRR Tolkien and EV Gordon 's Edition The poem in Middle
English Sir Gawain and the Green Knight public domain audiobook at LibriVox ( various translations ) Google Books results General information The Gawain / Pearl
Poet from the University of Calgary The Camelot Project Info on Sir Gawain Luminarium SGGK Website Oxford Bibliographies : Bibliography on Sir Gawain and the
Green Knight BBC documentary tour of locations thought to be found in the poem ... Holywell , the River Mersey , the ' Wild Wirral ' , the Peak District , the
Roaches , and Lud Chapel Sir Gawain Introduction - Article introducing various translations and adaptations of Sir Gawain The Pearl Poet Works Sir Gawain and
the Green Knight Pearl Patience Cleanness St.
</item 5>

<item 6>
Sir Gawain and the Green Knight
( 1980 ) . Christopher Tolkien , ed . Sir Gawain and the Green Knight , Pearl , and Sir Orfeo . Random House . ISBN 978 - 0 - 345 - 27760 - 2 . Jump up ^ White
, Michael ( 2003 ) . Tolkien : A Biography . New American Library . ISBN 0 - 451 - 21242 - 8 . Jump up ^ Farley , Frank E. ( 1930 ) . `` Rev. of Banks , Sir
Gawain , and Andrew , Sir Gawain '' ( PDF ) . Speculum. 5 ( 2 ) : 222 -- 24 . doi : 10.2307 / 2847870 . JSTOR 2847870 . Jump up ^ Baragona , Alan ( 2012 ) . ``
Rev. of Howes , Borroff , Sir Gawain and the Green Knight '' . Journal of English and Germanic Philology. 111 ( 4 ) : 535 -- 38 . doi : 10.5406 / jenglgermphil.
</item 6>

<item 7>
Sir Gawain and the Green Knight
JSTOR 462158 . Jump up ^ Arthur , pp. 121 -- 123 . Jump up ^ Frye , Northrop . Anatomy of Criticism , p. 186 . ISBN 0 - 691 - 01298 - 9 ^ Jump up to : Markman ,
Alan M. `` The Meaning of Sir Gawain and the Green Knight '' . PMLA . ( September 1957 ) 72.4 pp. 574 -- 586 Jump up ^ Sir Gawain and the Green Knight , second
edition . Ed . J.R.R. Tolkien and E.V. Gordon , 1925 , note to lines 2514ff . ASIN B000IPU84U Jump up ^ The Norton Anthology of English Literature , eighth
edition , Vol . BEd Stephen Greenblatt . New York , London : W.W. Norton and Co. , 2006 , p. 213 ( footnote ) . ISBN 0 - 393 - 92833 - 0 Jump up ^ Sir Gawain
and the Green Knight , line 2529 .
</item 7>

<item 8>
Sir Gawain and the Green Knight
doi : 10.5406 / jenglgermphil. 111.4. 0535 . Jump up ^ Hirsch , Edward ( 16 December 2007 ) . `` A Stranger in Camelot '' . The New York Times . p. 7.1 .
Retrieved 16 March 2010 . Jump up ^ Armitage , Simon ( 2007 ) . Sir Gawain and the Green Knight : A New Verse Translation . New York : Norton . ISBN 978 - 0 -
393 - 06048 - 5 . Jump up ^ Mills , Maldwyn ( 1991 ) . `` Review '' . The Yearbook of English Studies . 21 : 336 -- 337 . Jump up ^ `` Sir Gawain and the Green
Knight '' . BBC Four . 17 August 2010 . Retrieved 5 August 2012 . Jump up ^ Stone , Brian ( 1974 ) . Sir Gawain and the Green Knight . Penguin . p. 138 .
</item 8>

<item 9>
Sir Gawain and the Green Knight
Online : The Carle of Carlisle . Jump up ^ Kittredge , George Lyman . A Study of Sir Gawain and the Green Knight . Harvard University Press , 1960 . p. 76 .
ASIN B0006AWBXS Jump up ^ Kittredge , p. 83 Jump up ^ `` Sir Gawain and the Green Knight '' . Sir Gawain and the Green Knight , Pearl , Sir Orfeo . Translated
by J.R.R. Tolkien , New York : Ballantine Books , 1975 . 92 . Jump up ^ Burrow , J.A. A Reading of Sir Gawain and the Green Knight . London : Kegan Paul Ltd. ,
1965 . p. 162 . ISBN 0 - 7100 - 8695 - 4 ^ Jump up to : Burnley , J.D. ( 1973 ) . `` The Hunting Scenes in ' Sir Gawain and the Green Knight ' `` '' . The
Yearbook of English Studies . 3 : 1 -- 9 .
</item 9>

<item 10>
Sir Gawain and the Green Knight
Erkenwald Characters Green Knight Lady Bertilak Poetic style Bob and wheel Alliterative verse Wikiquote Wikisource texts ( hide ) Sir Gawain and the Green
Knight by Pearl Poet Characters Green Knight Gawain Lady Bertilak Guinevere Morgan le Fay Caradoc Film Gawain and the Green Knight ( 1973 ) Sword of the Valiant
( 1984 ) Literature The Squire , His Knight , and His Lady The Hollowing Opera Gawain Related The Greene Knight Honi soit qui mal y pense VIAF : 174581151 LCCN
: n79068347 GND : 4125015 - 1 SUDOC : 028245075 BNF : cb12012068r ( data ) Retrieved from `` https://en.wikipedia.org/w/index.
</item 10>
=== END ITEM id=492fe820dc8dc388 ===
