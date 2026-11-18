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

=== ITEM id=fb576ac445945ab9 ===
Gold sentences and facts:
Sentence 0: Each state controls where it administers its bar exam . Because the MBE ( below ) is a standardized test , it must be administered on the same day across the country . That day occurs twice a year as the last Wednesday in July and the last Wednesday in February . Two states , Delaware and North Dakota , may administer their bar exams only once , in July , if they do not have enough applicants to merit a second sitting . North Dakota requires ten applicants in order to administer the February exam . Most bar exams are administered on consecutive days . Louisiana is the exception , with the Louisiana Bar Exam being a three - day examination on Monday , Wednesday and Friday . Also , Louisiana 's examination is the longest in the country in terms of examination time , with seven hours on Monday and Wednesday and seven and one half hours on Friday for a total of 21.5 hours of testing . Montana 's bar examination also occurs over a three - day period , with a total of 18 hours of testing . The bar exams in Delaware , Nevada , Ohio , South Carolina , and Texas are also three days long . The California bar exam moved to a two - day format beginning with the July , 2017 exam .
  Fact 0.0: Each state controls where it administers its bar exam.
  Fact 0.1: The MBE is a standardized test.
  Fact 0.2: The MBE must be administered on the same day across the country.
  Fact 0.3: The MBE day occurs twice a year.
  Fact 0.4: The MBE day is the last Wednesday in July and the last Wednesday in February.
  Fact 0.5: Delaware and North Dakota may administer their bar exams only once, in July.
  Fact 0.6: Delaware and North Dakota may do this if they do not have enough applicants to merit a second sitting.
  Fact 0.7: North Dakota requires ten applicants in order to administer the February exam.
  Fact 0.8: Most bar exams are administered on consecutive days.
  Fact 0.9: Louisiana is the exception to consecutive-day bar exams.
  Fact 0.10: The Louisiana Bar Exam is a three-day examination on Monday, Wednesday and Friday.
  Fact 0.11: Louisiana's examination is the longest in the country in terms of examination time.
  Fact 0.12: Louisiana's examination has seven hours on Monday and Wednesday.
  Fact 0.13: Louisiana's examination has seven and one half hours on Friday.
  Fact 0.14: Louisiana's examination totals 21.5 hours of testing.
  Fact 0.15: Montana's bar examination occurs over a three-day period.
  Fact 0.16: Montana's bar examination totals 18 hours of testing.
  Fact 0.17: The bar exams in Delaware, Nevada, Ohio, South Carolina, and Texas are three days long.
  Fact 0.18: The California bar exam moved to a two-day format beginning with the July 2017 exam.
Sentence 1: To refresh their memory on `` black - letter rules '' tested on the bar , most students engage in a regimen of study ( called `` bar review '' ) between graduating from law school and sitting for the bar . For bar review , most students in the United States attend a private bar review course which is provided by a third - party company and not their law school .
  Fact 1.0: Most students engage in a regimen of study called bar review.
  Fact 1.1: Bar review is done to refresh memory on black-letter rules tested on the bar.
  Fact 1.2: Bar review is done between graduating from law school and sitting for the bar.
  Fact 1.3: For bar review, most students in the United States attend a private bar review course.
  Fact 1.4: The private bar review course is provided by a third-party company.
  Fact 1.5: Most students in the United States do not take their bar review course from their law school.

Retrieved items:
<item 1>
Bar examination
Criminal law , employment law , constitution law , administrative law , and tax law . Civil procedure law , bankruptcy and business reorganization law , and the
system of the court of justice . And Criminal procedure law , human rights , and law on the evidence . Each part has 10 essay questions . The pass mark is 50 .
The parts 1 - 2 are usually taken in October and the rest are usually taken in March . One does not need to pass all four parts in one year . After passing all
the written exams , there is an oral exam . Around 10,000 bar students sit the exam each year . In 2013 , 1,231 students were called to the bar , 111 of whom
did it in only one year . Quite confusingly with international norms , students called to the bar are referred to as netibandit ( เนติบัณฑิต ) , which gets
translated into English as Barrister - at - Law . The Thai legal profession , however , is a fused one and those with lawyer licenses are able to practice both
as barristers and solicitors in the British / Commonwealth sense .
</item 1>

<item 2>
Bar examination
The common law provinces all require prospective lawyers to complete a term of articles ( usually 10 months ) after graduation from law school during which they
work under the supervision of a qualified lawyer . The bar exams may be taken after graduation from law school , but before the commencement of articling , or
may be taken during or after articling is completed . Once the barristers ' and solicitors ' exams have been passed and the term of articles is successfully
completed , students may then be called the bar and admitted to the legal profession as lawyers ( barristers and solicitors ) . People 's Republic of China (
edit ) Main article : State Judicial Exam England and Wales ( edit ) Since the UK has a separated legal profession , Law graduates in England and Wales can take
examinations to qualify as a Barrister or a Solicitor by either undertaking the BPTC or the LPC respectively . These courses are the vocational part of the
training required under the Bar Association and The Law Society rules and are either undertaken on a full - time basis for one year or on a part - time basis
over two years .
</item 2>

<item 3>
Bar examination
After successfully completing these courses , which generally include various examinations and practical ability tests , graduates must secure either a Training
Contract ( for those who have completed the LPC ) or a Pupillage ( for those who have completed the BPTC ) . These are akin to Articling positions in other
jurisdictions and are the final Practical stage before being granted full admission to practice . The general timescale therefore to become fully qualified
after entering Law School can range between 6 -- 7 years ( assuming no repeats are required ) . However some controversy remains about the lack of Training
Contracts and Pupillages available to graduates even after having completed the LPC / BPTC . These courses can vary in cost anywhere from £ 9,000 to £ 17,000
and are generally undertaken by students on a private basis making them incur additional costs . The final debt in student fees alone after having completed the
academic and vocational training can range between £ 20,000 - £ 25,000 . This is set to increase to £ 40,000 - £ 50,000 for students entering law school in the
years 2012 due to the increased tuition fees for Law School itself .
</item 3>

<item 4>
Bar examination
This exam is composed of three parts : Criminal Law , Criminal Procedural Law and Law of Criminal Enforcement Civil Law , Civil Procedural Law and Business Law
Constitutional Law , Administrative Law , Labor Law , Social Security Law and Law of the European Union . After passing these exams the candidate can practice
law as an attorney - at - law / barrister or as a secretary / judge at the court or as a prosecutor at the public prosecutor 's office or in - house legal
counsel or may operate individually at any field of law . Ireland ( edit ) The bar exams in Ireland are the preserve of the Honorable Society of King 's Inns ,
which runs a series of fourteen exams over ten weeks , from March to June each year , for those enrolled as students in its one - year Barrister - at - Law
degree course . These exams cover such skills as advocacy , research and opinion writing , consulting with clients , negotiation , drafting of legal documents
and knowledge of civil and criminal procedure . For those who fail to meet the requisite 50 % pass mark , repeats are held in the following August and September
.
</item 4>

<item 5>
Bar examination
To be called to the Singaporean Bar , all law graduates must complete the following : Sit for , and pass the Singapore Bar Examinations ( Part B) . Completed a
six - month training contract known as the Practice Training Contract with a local law firm , or practice . Subject to an order from Court , legal trainees who
have completed at least three months of their Practice Training Contract , as well as sat for and passed the Singapore Bar Examinations ( Part B ) , are able to
seek partial admission to the Bar . By applying to the Court to be part - called , candidates are granted the right to appear before the Court in specific
circumstances . In addition , all law graduates of non-Singaporean university must complete the following additional requirements : Sit for , and pass the
Singapore Bar Examinations ( Part A ) . The five subjects which are offered in this exam are , Administrative and Constitutional Law , Company Law , Criminal
Law , Evidence , and Land Law . All foreign university graduates are required to pass this exam to prove competence in Singaporean law . Part A of the Bar
Examinations are administered by SILE as well .
</item 5>

<item 6>
Bar examination
Because the MBE ( below ) is a standardized test , it must be administered on the same day across the country . That day occurs twice a year as the last
Wednesday in July and the last Wednesday in February . Two states , Delaware and North Dakota , may administer their bar exams only once , in July , if they do
not have enough applicants to merit a second sitting . North Dakota requires ten applicants in order to administer the February exam . Most bar exams are
administered on consecutive days . Louisiana is the exception , with the Louisiana Bar Exam being a three - day examination on Monday , Wednesday and Friday .
Also , Louisiana 's examination is the longest in the country in terms of examination time , with seven hours on Monday and Wednesday and seven and one half
hours on Friday for a total of 21.5 hours of testing . Montana 's bar examination also occurs over a three - day period , with a total of 18 hours of testing .
The bar exams in Delaware , Nevada , Ohio , South Carolina , and Texas are also three days long . The California bar exam moved to a two - day format beginning
with the July , 2017 exam .
</item 6>

<item 7>
Bar examination
A logbook signed by the judge on the bench has to certify their weekly attendance . By the end of the eighteenth month , they are eligible to apply to take the
Final Bar Exam by submitting their case summaries , the logbook and a research work pre-approved by the Bar . It is noteworthy , however , that during these
eighteen months , Trainees are eligible to have a limited practice of law under the supervision of their supervising Attorney . This practice does not include
Supreme Court eligible cases and certain criminal and civil cases . Candidates will be tested on Civil law , Civil Procedure , Criminal law , Criminal Procedure
, Commercial Law , Notary ( including rules pertaining Official Documents , Land & Real Estate registrations and regulations etc . ) . Each exam takes two days
, a day on oral examination in front of a judge or an attorney , and a day of essay examination , in which they will be tested on hypothetical cases submitted
to them . Successful applicants will be honoured with the title of `` First Degree Attorney '' , after they take the oath and can practice in all courts of the
country including the Supreme Court . Those who fail must redo the apprenticeship program in full or in part before taking the Final Bar Exam again .
</item 7>

<item 8>
Bar examination
The MEE and MPT , as uniform though not standardized tests , also must be administered on the same day across the country -- specifically the day before the MBE
. Preparation for the exam ( edit ) Most law schools teach students common law and how to analyze hypothetical fact patterns like a lawyer , but do not
specifically prepare law students for any particular bar exam . Only a minority of law schools offer bar preparation courses . To refresh their memory on ``
black - letter rules '' tested on the bar , most students engage in a regimen of study ( called `` bar review '' ) between graduating from law school and
sitting for the bar . For bar review , most students in the United States attend a private bar review course which is provided by a third - party company and
not their law school . Multistate standardized Examinations ( edit ) The National Conference of Bar Examiners ( NCBE ) is a U.S. based non-profit organization
that develops national ( `` multistate '' ) standardized tests for admission to the bar in individual states . The organization was founded in 1931 .
</item 8>

<item 9>
Bar examination
The exam is divided in two stages -- the first consists of 80 multiple choice questions covering all disciplines ( Ethics , Human Rights , Philosophy of Law ,
Constitutional Law , Administrative Law , Civil Law , Consumer Law , Civil Procedure Law , Criminal Law , Criminal Procedure Law , Labour Law and Labour
Procedure Law ) . The candidate must score at least 40 questions correctly to proceed to the second part of the exam , four essay questions and a drafting
project ( motion , opinion or claim document ) in Civil Law ( including Consumer Law ) , Labour Law , Criminal Law , Administrative Law , Constitutional Law ,
Corporate Law or Tax Law , and their respective procedures . The Bar examination can be taken on the graduation year . Success in the examination allows one to
practice in any court or jurisdiction of the country . Canada ( edit ) In Canada , admission to the bar is a matter of provincial or territorial jurisdiction .
All provinces except for Québec follow a common law tradition . Lawyers in every common law province are qualified as both barristers and solicitors , and must
pass a Barristers ' Examination and Solicitors ' Examination administered by the Law Society that governs the legal profession in their respective province or
territory .
</item 9>

<item 10>
Bar examination
The new law school system that began in 2009 allows only the graduates of a law school to apply for the bar exam . Malaysia ( edit ) Main article : Certificate
in Legal Practice ( Malaysia ) Philippines ( edit ) The Philippine Bar Examination is administered once every year on the four Sundays of November ( September
before 2011 ) . It covers eight areas of law : political law , labor law and social legislation , criminal law , civil law , commercial law , taxation law ,
remedial law , and legal ethics and practical exercises . Poland ( edit ) In Poland , the bar examination is taken after graduating from a law faculty at a
university . It allows a person to undertake practice , the duration of which varies depending on the specialization . After the practical period applicants
must pass the exam held by the Professional Chambers with assistance from some members of the Ministry of Justice . Singapore ( edit ) The Singapore Bar
Examination ( Part B) is administered once every year , usually over the course of four days . The exam is generally held on the last week of November , and is
administered by the Singapore Institute of Legal Education ( SILE ) .
</item 10>
=== END ITEM id=fb576ac445945ab9 ===

=== ITEM id=024723e661e6bb42 ===
Gold sentences and facts:
Sentence 0: R.H. Thomson as Matthew Cuthbert
  Fact 0.0: R.H. Thomson played the role of Matthew Cuthbert.

Retrieved items:
<item 1>
Anne (TV series)
Anne ( TV series ) - wikipedia Anne ( TV series ) Jump to : navigation , search Anne Also known as Anne with an E Genre Drama Created by Moira Walley - Beckett
Based on Anne of Green Gables by Lucy Maud Montgomery Screenplay by Moira Walley - Beckett Starring Amybeth McNulty Geraldine James R.H. Thomson Lucas Jade
Zumann Dalila Bela Corrine Koslo Aymeric Jett Montaz Opening theme `` Ahead by a Century '' by The Tragically Hip Country of origin Canada Original language ( s
) English No. of seasons No.
</item 1>

<item 2>
Anne (TV series)
Jump up ^ `` Anne with an E : Season 1 '' . Metacritic . Retrieved 17 June 2017 . Jump up ^ Ashby , Emily . `` Anne with an E '' . Common Sense Media .
Retrieved 17 June 2017 . Jump up ^ Cerny , Tasha . `` ANNE WITH AN E Review : `` Episodes 1 - 3 '' `` . www.thetrackingboard.com . Retrieved 18 June 2017 . Jump
up ^ Inhat , Gwen . `` Anne With An E offers a winning , darker take on a familiar tale '' . A.V. Club . Retrieved 18 June 2017 . Jump up ^ Schneller , Johanna
( March 20 , 2017 ) . `` This Anne of Green Gables hints at miseries : What I 'm Watching '' . Toronto Star . access - date = requires url = ( help ) Jump up ^
Nguyen , Hanh . `` ' Anne with an E ' Review : A ' Breaking Bad ' Producer Updates ' Anne of Green Gables ' for Our Darker Times '' .
</item 2>

<item 3>
Anne (TV series)
7 `` Wherever You Are Is My Home '' Amanda Tapping Moira Walley - Beckett April 30 , 2017 ( 2017 - 04 - 30 ) N / A The Cuthberts vow to do whatever it takes to
save the farm which reminds Anne of the strength of friendship and love . Reception ( edit ) Critical response ( edit ) The series has achieved a rating of 88 %
on Rotten Tomatoes , the site 's critical consensus stating : `` Anne with an E uses its complex central character to offer a boldly stylish , emotionally
resonant spin on classic source material that satisfies in its own right . '' The series has received a rating of 79 on Metacritic based on fifteen reviews ,
indicating `` generally favorable reviews '' . Emily Ashby , writing for Common Sense Media , calls the series an `` exceptional '' and `` spectacular ''
interpretation , giving it four out of five stars . Tasha Cerny , contributor for the Tracking Board , praises the cinematography as lush and colourful , the
characters vibrant , and the plot `` surprisingly thrilling for a story about a young girl living in a small secluded community in the late nineteenth century .
</item 3>

<item 4>
Anne (TV series)
`` CBC and Netflix renew reimagined Anne of Green Gables '' . Toronto Star . Retrieved August 3 , 2017 . Jump up ^ `` Anne ( with an E ) : Season Two Filming
Begins for CBC and Netflix Series '' . TV Series Finale . November 24 , 2017 . ^ Jump up to : `` Netflix 's ' Anne of Green Gables ' Adaptation Finds Its Anne
Shirley , Casts 2 Other Roles '' . Deadline . October 21 , 2016 . ^ Jump up to : `` Netflix 's ' Anne Of Green Gables ' Adaptation Adds Three To Cast '' .
Deadline.com . Retrieved November 21 , 2016 . Jump up ^ `` Maureen McCormick To Guest In ' The Guest Book ' ; Lucas Jade Zumann Joins ' Anne ' '' . Deadline.com
. Retrieved November 18 , 2016 . Jump up ^ Hunt , Nigel ( March 19 , 2017 ) . `` ' Ai n't your grandmother 's Anne ' : new series gives gritty Green Gables amid
glut of Anne adaptations '' . CBC News .
</item 4>

<item 5>
Anne (TV series)
Writing for Entertainment Weekly , Isabella Beidenharn expressed similar feelings , but , `` putting the source material aside , it 's a fine show on its own ''
, and she conceded that `` inventing a dark side might help Anne With an E fit into today 's TV landscape '' . Allison Keene , writing for Collider , agrees
that Anne is a good drama on its own terms , but allows it is `` only a fair adaptation '' of the novel , at its best in the home scenes : `` Anne with an E is
undeniably the most stylish adaptation we 've ever seen of Anne of Green Gables . But its desire to reveal more of Anne 's miserable past in order to be more
true to what the desperation of an orphan is like feels at odds with Montgomery 's story . ''
</item 5>

<item 6>
Anne (TV series)
Numeris . Jump up ^ Brioux , Bill ( @ BillBriouxTV ) ( April 10 , 2017 ) . `` SUN onites SNET NHL TOR - COL 1181k VAN -- EDM 766k TSN CURL 1026k GLO NCISLA 1Mk
CBC ANNE 654k STORY US 546k CITY SHOTS 623k CTV SAVE HOPE 548k '' ( Tweet ) -- via Twitter . Jump up ^ Brioux , Bill ( @ BillBriouxTV ) ( April 24 , 2017 ) . ``
SJN o'nites SNet NHL TOR - WASH Game 6 3467k OTT - BOS 1284k OMNI2 TOR - WASH 121k CTV SAVE HOPE 686K CBC ANNE 656K STORY OF US 450K '' ( Tweet ) -- via Twitter
. Jump up ^ `` Anne with an E ( 2017 - ) '' . Rotten Tomatoes . Retrieved 4 January 2018 . Jump up ^ `` Anne with an E : Season 1 '' . Metacritic .
</item 6>

<item 7>
Anne (TV series)
of seasons No. of episodes 7 ( list of episodes ) Production Executive producer ( s ) Elizabeth Bradley Alex Sapot Sally Catto Debra Hayward Alison Owen Miranda
de Pencier Moira Walley - Beckett Producer ( s ) Susan Murdoch Cinematography Bobby Shore Running time 44 minutes Production company ( s ) Pelican Ballet
Northwood Entertainment Distributor Netflix Release Original network CBC Television Netflix ( worldwide ) Picture format 4K ( Ultra HD ) Audio format Dolby
Digital 5.1 with Descriptive Video Service track Original release March 19 , 2017 ( 2017 - 03 - 19 ) -- present ( present ) External links Website Anne ( titled
Anne with an E for international distribution on Netflix ) is a Canadian drama television series based on the 1908 novel Anne of Green Gables by Lucy Maud
Montgomery , and adapted by Emmy Award - winning writer and producer Moira Walley - Beckett . It airs on CBC in Canada , and elsewhere in the world it is
available for streaming on Netflix . The first season consists of seven episodes , with Niki Caro directing the 90 - minute season premiere .
</item 7>

<item 8>
Anne (TV series)
Casting ( edit ) Approximately 1800 girls on three continents auditioned for the role of Anne Shirley . Amybeth McNulty was chosen for her ability to deliver
dialogue which is `` incredibly thick and dynamic and beautiful '' , according to Miranda de Pencier . Walley - Beckett describes her as at once `` luminous ,
'' transparent , smart , soulful and emotional . Filming locations ( edit ) The series is filmed partially in Prince Edward Island as well as locations in
Southern Ontario ( including Millbrook and Caledon ) . Music ( edit ) The opening theme is the song `` Ahead by a Century '' performed and originally composed
by Canadian band The Tragically Hip . Themes ( edit ) Moira Walley - Beckett had this to say about her treatment , which is darker than the previous productions
: `` In this day and age , themes of identity , prejudice , bullying , being an outsider , searching for a way to be accepted and how to belong are entirely
topical and super relevant , and those are themes that are built into the story of ' Anne . '
</item 8>

<item 9>
Anne (TV series)
'' She went on to call Anne Shirley an `` accidental feminist '' , and how she `` really wanted to tell this story now . '' Broadcast and Release ( edit ) The
series initially premiered on March 19 , 2017 , on CBC and aired on a weekly basis , the season finale airing on April 30 , 2017 . The series debuted on Netflix
on May 12 , 2017 , under the title Anne with an E . Episodes ( edit ) Season 1 ( 2017 ) ( edit ) No . Title Directed by Written by Original air date Can .
viewers ( millions ) `` Your Will Shall Decide Your Destiny '' Niki Caro Moira Walley - Beckett March 19 , 2017 ( 2017 - 03 - 19 ) 0.999 When a miscommunication
brings a girl , Anne Shirley , to Green Gables instead of a boy , Marilla and Matthew Cuthbert are faced with a life - changing decision .
</item 9>

<item 10>
Anne (TV series)
When Matthew goes to pick the child up at the railway station , he finds not a boy , but a high - spirited and talkative girl , Anne Shirley . At first , the
Cuthberts are inclined to send her back , particularly after Marilla 's brooch goes missing , and Anne , in despair , runs away . Reconciled and settled in her
new home and school , Anne once again displays boundless enthusiasm which is nevertheless easily turned into despair when things go wrong , which they often do
. Slowly , her ebullient nature wins over those around her . Cast ( edit ) Main ( edit ) Amybeth McNulty as Anne Shirley Geraldine James as Marilla Cuthbert
R.H. Thomson as Matthew Cuthbert Dalila Bela as Diana Barry Lucas Jade Zumann as Gilbert Blythe Corrine Koslo as Rachel Lynde Aymeric Jett Montaz as Jerry
Baynard Others ( edit ) Jonathan Holmes as Mr. William Barry Helen Johns as Mrs. Eliza Barry Ryan Kiera Armstrong as Minnie Mary Barry Deborah Grover as
Josephine Barry Wayne Best as John Blythe Phillip Williams as Thomas Lynde David Ingram as Mr.
</item 10>
=== END ITEM id=024723e661e6bb42 ===

=== ITEM id=426cf59543d031ae ===
Gold sentences and facts:
Sentence 0: The Curse of Oak Island follows brothers Marty and Rick Lagina , originally from Kingsford , Michigan , through their efforts to find the speculated treasure or historical artifacts believed to be on Oak Island . The series discusses the history of the island , recent discoveries , theories , and prior attempts to investigate the site . Areas of interest include the `` Money Pit '' , Borehole 10 - x , Smith 's Cove , `` Nolan 's Cross '' , the `` Hatch '' , the `` Watchtower '' and the `` Swamp '' .
  Fact 0.0: The Curse of Oak Island is a series.
  Fact 0.1: The Curse of Oak Island follows brothers Marty and Rick Lagina.
  Fact 0.2: Marty and Rick Lagina are originally from Kingsford, Michigan.
  Fact 0.3: The Curse of Oak Island follows the Laginas' efforts to find the speculated treasure or historical artifacts believed to be on Oak Island.
  Fact 0.4: The series discusses the history of Oak Island.
  Fact 0.5: The series discusses recent discoveries on Oak Island.
  Fact 0.6: The series discusses theories about Oak Island.
  Fact 0.7: The series discusses prior attempts to investigate the site on Oak Island.
  Fact 0.8: Areas of interest in the series include the "Money Pit".
  Fact 0.9: Areas of interest in the series include Borehole 10-x.
  Fact 0.10: Areas of interest in the series include Smith's Cove.
  Fact 0.11: Areas of interest in the series include "Nolan's Cross".
  Fact 0.12: Areas of interest in the series include the "Hatch".
  Fact 0.13: Areas of interest in the series include the "Watchtower".
  Fact 0.14: Areas of interest in the series include the "Swamp".

Retrieved items:
<item 1>
The Curse of Oak Island
Contents ( hide ) 1 Overview 1.1 Background 1.2 Theories 2 Series overview 3 Production 4 Episodes 4.1 Season 1 ( 2014 ) 4.2 Season 2 ( 2014 -- 2015 ) 4.3
Season 3 ( 2015 -- 2016 ) 4.4 Season 4 ( 2016 -- 2017 ) 4.5 Season 5 ( 2017 -- 2018 ) 5 References 6 Further reading 7 External links Overview ( edit ) The
Curse of Oak Island follows brothers Marty and Rick Lagina , originally from Kingsford , Michigan , through their efforts to find the speculated treasure or
historical artifacts believed to be on Oak Island . The series discusses the history of the island , recent discoveries , theories , and prior attempts to
investigate the site . Areas of interest include the `` Money Pit '' , Borehole 10 - x , Smith 's Cove , `` Nolan 's Cross '' , the `` Hatch '' , the ``
Watchtower '' and the `` Swamp '' .
</item 1>

<item 2>
The Curse of Oak Island
S. Wild West Tech The XY Factor Upcoming Blue Book The Commanders Retrieved from ``
https://en.wikipedia.org/w/index.php?title=The_Curse_of_Oak_Island&oldid=832882458 '' Categories : 2014 American television series debuts History ( U.S. TV
channel ) original programs English - language television programs 2010s American television series Television shows set in Nova Scotia Oak Island Hidden
categories : Use mdy dates from August 2014 Talk Contents About Wikipedia Français Latviešu Română Русский Edit links This page was last edited on 28 March 2018
, at 14 : 10 . About Wikipedia
</item 2>

<item 3>
The Curse of Oak Island
overall No. in season Title Original air date Viewers ( millions ) 29 `` Going for Broke '' November 15 , 2016 ( 2016 - 11 - 15 ) 2.62 The team agrees to turn
focus over to the Money Pit and review what they have collected . Researcher Zena Halpern presents a handmade copy of what she claims is a French map from 1347
of Oak Island . The map shows locations for a `` hatch '' , a `` valve '' and an `` anchor '' on the Island . A theory is presented based on a purported
decoding of a cipher that Joab at the time of King David transported the Ark of the Covenant to North Africa from which it was later transported onward by the
Knights Templar from the then - Portuguese town of Safi in Morocco to Oak Island . 30 `` Always Forward '' November 22 , 2016 ( 2016 - 11 - 22 ) 2.66 While
preparing to excavate the Money Pit , the team investigates an 18th century well on mainland Nova Scotia .
</item 3>

<item 4>
The Curse of Oak Island
The Curse of Oak Island - wikipedia The Curse of Oak Island Jump to : navigation , search The Curse of Oak Island Genre Reality Narrated by Robert Clotworthy
Country of origin Canada Original language ( s ) English No. of seasons 5 No. of episodes 62 Production Executive producer ( s ) Kevin Burns Camera setup
Multiple Running time 43 minutes Production company ( s ) Prometheus Entertainment Distributor A&E Networks Release Original network History Canada Original
release January 5 , 2014 ( 2014 - 01 - 05 ) -- present External links Website Production website The Curse of Oak Island is an active reality television series
that first premiered in Canada on the History network on January 5 , 2014 . The show features what is known as the Oak Island mystery , showing efforts to
search for historical artifacts and treasure .
</item 4>

<item 5>
The Curse of Oak Island
2018 '' . Showbuzz Daily . Retrieved March 7 , 2018 . Further reading ( edit ) MacDonald , David ( 1965 ) . `` Treasure Hunt : The Mystery of Oak Island '' .
Reader 's Digest -- via readersdigest.ca .
</item 5>

<item 6>
The Curse of Oak Island
61 17 `` A Family Album '' February 27 , 2018 ( 2018 - 02 - 27 ) 1.98 The episode includes accounts of activities on the Oak Island by the McGinnis family ,
descendants of Samuel Ball , William Chappell 's family , the Restall Family and the family of Robert Dunfield . The Blankenship / Nolan disputes on Oak Island
are discussed . The episode ends with a short history of the Lagina brothers ' interest in Oak Island . Note : This episode is a special . 62 18 `` Amazing
Discoveries '' March 6 , 2018 ( 2018 - 03 - 06 ) 2.89 The gem in the Jewelry fragment found on the island is a Rhodolite Garnet stone of small value . The
drilling obstruction in the `` money pit '' borehole is determined by divers to be a granite boulder . References ( edit ) ^ Jump up to : `` Shows A-Z - curse
of oak island , the on history '' . The Futon Critic . Retrieved December 6 , 2017 .
</item 6>

<item 7>
The Curse of Oak Island
Background ( edit ) The Lagina brothers became fascinated with the island after reading the January 1965 issue of Reader 's Digest magazine that features an
article on the Restall family 's work to investigate the so - called `` Money Pit '' . Marty and his brother Rick obtained a controlling interest in Oak Island
Tours , which reportedly owns most of the island . The brothers were later approached by Prometheus Entertainment to do a reality show . Rick and Marty have
engaged the assistance of father - and - son Dan and Dave Blankenship , permanent residents of the island who have likewise been searching for treasure since
the 1960s . Theories ( edit ) The series explores various Oak Island theories and conjectures through conversations with independent researchers .
</item 7>

<item 8>
Graph query row:
filming_location: Nova Scotia
island: Oak Island
island_location: None
</item 8>

<item 9>
The Curse of Oak Island
Series overview ( edit ) Season Episodes Originally aired First aired Last aired 5 January 5 , 2014 ( 2014 - 01 - 05 ) February 9 , 2014 ( 2014 - 02 - 09 ) 10
November 4 , 2014 ( 2014 - 11 - 04 ) January 13 , 2015 ( 2015 - 01 - 13 ) 13 November 10 , 2015 ( 2015 - 11 - 10 ) February 2 , 2016 ( 2016 - 02 - 02 ) 16
November 15 , 2016 ( 2016 - 11 - 15 ) February 21 , 2017 ( 2017 - 02 - 21 ) 5 18 November 7 , 2017 ( 2017 - 11 - 07 ) March 6 , 2018 ( 2018 - 03 - 06 )
Production ( edit ) Having started in 2014 , The Curse of Oak Island is currently in its fifth season . On August 2 , 2016 , it was announced that Nova Scotia
Business Inc.
</item 9>

<item 10>
The Curse of Oak Island
On August 2 , 2016 , it was announced that Nova Scotia Business Inc. approved $1,271,546 in film funding for the production of the 4th season . In October 2017
, a fifth season was announced , which began airing on November 7 , 2017 . Episodes ( edit ) Season 1 ( 2014 ) ( edit ) No . overall No. in season Title
Original air date Viewers ( millions ) `` What Lies Below '' January 5 , 2014 ( 2014 - 01 - 05 ) 2.53 Brothers Rick and Marty Lagina now own most of Oak Island
. Determined to resume the search for treasure on the island , they explore an abandoned shaft called Borehole 10 - X and use water pressure to pump material to
the surface . They find a tiny piece of metal .
</item 10>
=== END ITEM id=426cf59543d031ae ===

=== ITEM id=276a19b86d540f21 ===
Gold sentences and facts:
Sentence 0: A main vertical stroke is called a stem . The letter m has three , the left , middle , and right stems . The central stroke of an s is called the spine . When the stroke is part of a lowercase and rises above the height of an x ( called the x height ) , it is called an ascender . Letters with ascenders are bdfhkl . A stroke which drops below the baseline is a descender . Letters with descenders are gjpqy . An arching stroke is called a shoulder as in the top of an R or sometimes just an arch , as in hnm . A closed curved stroke is called a bowl in bdopq DOPQ ; B has two bowls . A trailing outstroke , as in jy JQR is called a tail . The inferior diagonal stroke in K is called leg . A short horizontal stroke , as in the center of ef and the middle stroke of EF , is called a bar . When the strokes connect as in A and H or cross strokes as in t is also known as crossbar . A longer horizontal stroke at the top or bottom , as in ET , is called an arm . The bottom of the two - story g is called a loop ; the very short stroke at the top is called the ear . ij each have a dot , jot , or tittle . Angles of strokes are called apices if at the top and vertices if at the bottom . w has one apex and two vertices ; v has one vertex .
  Fact 0.0: A main vertical stroke is called a stem.
  Fact 0.1: The letter m has three stems: the left stem, the middle stem, and the right stem.
  Fact 0.2: The central stroke of an s is called the spine.
  Fact 0.3: When a stroke is part of a lowercase letter and rises above the height of an x (called the x height), it is called an ascender.
  Fact 0.4: The letters with ascenders are b, d, f, h, k, and l.
  Fact 0.5: A stroke which drops below the baseline is called a descender.
  Fact 0.6: The letters with descenders are g, j, p, q, and y.
  Fact 0.7: An arching stroke is called a shoulder, as in the top of an R.
  Fact 0.8: An arching stroke is sometimes just called an arch, as in h, n, and m.
  Fact 0.9: A closed curved stroke is called a bowl, as in b, d, o, p, q, D, O, P, and Q.
  Fact 0.10: The letter B has two bowls.
  Fact 0.11: A trailing outstroke, as in j, y, J, Q, and R, is called a tail.
  Fact 0.12: The inferior diagonal stroke in K is called a leg.
  Fact 0.13: A short horizontal stroke, as in the center of e and f and the middle stroke of E and F, is called a bar.
  Fact 0.14: When strokes connect, as in A and H, or cross, as in t, the stroke is also known as a crossbar.
  Fact 0.15: A longer horizontal stroke at the top or bottom, as in E and T, is called an arm.
  Fact 0.16: The bottom of the two-story g is called a loop.
  Fact 0.17: The very short stroke at the top of the two-story g is called the ear.
  Fact 0.18: The letters i and j each have a dot, jot, or tittle.
  Fact 0.19: Angles of strokes are called apices if at the top.
  Fact 0.20: Angles of strokes are called vertices if at the bottom.
  Fact 0.21: The letter w has one apex and two vertices.
  Fact 0.22: The letter v has one vertex.

Retrieved items:
<item 1>
Church Fathers
549 ) Jump up ^ Herbermann , Charles , ed. ( 1913 ) . `` Tertullian '' . Catholic Encyclopedia . New York : Robert Appleton Company . Jump up ^ A History of
Christian Thought , Paul Tillich , Touchstone Books , 1972 . ISBN 0 - 671 - 21426 - 8 ( p. 43 ) Jump up ^ To Autolycus , Book 2 , chapter XV Jump up ^ Known in
Latin and Low Franconian as Ambrosius , in Italian as Ambrogio and in Lombard as Ambroeus . Jump up ^ Cross , F.L. , ed . The Oxford Dictionary of the Christian
Church . New York : Oxford University Press . 2005 , article Platonism Jump up ^ Pope St. Gregory I at about.com Jump up ^ Karim , Cyril Aphrem ( December 2004
) . Symbols of the cross in the writings of the early Syriac Fathers . Gorgias Press LLC . p. 3 . ISBN 978 - 1 - 59333 - 230 - 3 .
</item 1>

<item 2>
The Curse of Oak Island
They find a carving of a multi-armed cross and offer an unsupported theory that it is a representation of the `` Tree of Life '' and that it is the same as
Nolan 's Cross on the island . A theory is presented that the cave carving may represent an `` earlier version '' of the `` Tree of Life '' . Based on the
information discovered in France , the team concludes that the Oak Island swamp should be excavated again . 54 10 `` The Signs Of A Cross '' January 16 , 2018 (
2018 - 01 - 16 ) 3.34 Eight pages of what is alleged to be a transcript of an incomplete log of an unnamed ship are presented . It is claimed , without any
evidence , that the log is from a French ship which participated in the Duc d'Anville Nova Scotia expedition of 1746 . More wood , a bone and a metal spike are
brought up from a borehole in the money pit area . The shaft excavation reaches an obstruction at 170 feet . A small lead cross is found at Smith 's Cove .
</item 2>

<item 3>
Typeface anatomy
The central stroke of an s is called the spine . When the stroke is part of a lowercase and rises above the height of an x ( called the x height ) , it is
called an ascender . Letters with ascenders are bdfhkl . A stroke which drops below the baseline is a descender . Letters with descenders are gjpqy . An arching
stroke is called a shoulder as in the top of an R or sometimes just an arch , as in hnm . A closed curved stroke is called a bowl in bdopq DOPQ ; B has two
bowls . A trailing outstroke , as in jy JQR is called a tail . The inferior diagonal stroke in K is called leg . A short horizontal stroke , as in the center of
ef and the middle stroke of EF , is called a bar . When the strokes connect as in A and H or cross strokes as in t is also known as crossbar . A longer
horizontal stroke at the top or bottom , as in ET , is called an arm . The bottom of the two - story g is called a loop ; the very short stroke at the top is
called the ear .
</item 3>

<item 4>
Typeface anatomy
`` Typeface Mechanics : 001 '' . Frere - Jones Type . Retrieved 17 April 2016 . Frere - Jones , Tobias . `` Typeface Mechanics : 002 '' . Frere - Jones Type .
Retrieved 17 April 2016 . Coles , Stephen . `` The Anatomy of Type '' . Blog for Coles ' 2012 book The Anatomy of Type : A Graphic Guide to 100 Typefaces . ``
Glossary '' . FontShop . Monotype . ( hide ) Typography terminology Page Canons of page construction Column Even working Margin Page numbering Pagination Pull
quote Recto and verso Paragraph Alignment Justification Leading River Sentence spacing Widows and orphans Character Typeface anatomy Counter Diacritics Dingbat
Glyph Initial Kerning Letter - spacing Ligature Subscript and superscript Swash Text figures Tittle Capitalization ALL CAPS Camel Case Letter case Petite caps
Small caps Visual distinction Italics Oblique Bold Color Underline Blackboard bold Blackletter Infɑnt Vertical aspects Ascender Baseline Cap height Descender
Median Overshoot x-height Classifications Roman type Antiqua ( old style ) Didone ( modern ) Sans - serif Script Serif Slab serif Transitional Reverse
</item 4>

<item 5>
Typeface anatomy
Typeface Anatomy - wikipedia Typeface Anatomy Jump to : navigation , search Typographic parts of a glyph : 1 ) x-height ; 2 ) ascender line ; 3 ) apex ; 4 )
baseline ; 5 ) ascender ; 6 ) crossbar ; 7 ) stem ; 8 ) serif ; 9 ) leg ; 10 ) bowl ; 11 ) counter ; 12 ) collar / link / neck ; 13 ) loop ; 14 ) ear ; 15 ) tie
; 16 ) horizontal bar ; 17 ) arm ; 18 ) vertical bar ; 19 ) cap height ; 20 ) descender line . Typeface anatomy describes the graphic elements that make up
printed letters in a typeface . Contents ( hide ) 1 Strokes 2 Terminals 3 Space 4 Proportions 5 Metal type era 6 See also 7 References 8 Further reading 9
External links Strokes ( edit ) The strokes are the components of a letterform . Strokes may be straight , as in klvwxz , or curved , as in cos . If straight ,
they may be horizontal , vertical , or diagonal ; if curved , open or closed .
</item 5>

<item 6>
Typeface anatomy
ij each have a dot , jot , or tittle . Angles of strokes are called apices if at the top and vertices if at the bottom . w has one apex and two vertices ; v has
one vertex . The font shown in the example is stressed : strokes have varying widths . In this example , the stroke at the top of the g is thinner at the top
and bottom than on the sides -- a vertical stress . Terminals ( edit ) The terminals ( ends ) of instrokes and outstrokes often end in serif or sans - serif . A
seriffed terminal may be described as a wedge , bulbous , teardrop , slab , etc. , depending on the design of the type . Sorting typefaces can be practical
based on its look . A clue feature is to define the serif style . Some designs also have spurs , which are smaller than serifs and appear on angles rather than
at a terminal , as on e or G . Space ( edit ) Areas of negative space ( white space ) formed by straight or curved strokes are called counters . Closed counters
are found in abdegopq ABDOPQR , and open counters in acefhmnrstu .
</item 6>

<item 7>
Typeface anatomy
Harper Design , 2013 . ^ Jump up to : Dean , Paul ( 27 March 2008 ) . `` eXtreme Type Terminology . Part 2 : Anatomy of a Letterform '' . I Love Typography .
Further reading ( edit ) Gaskell , Philip ( January 1976 ) . `` A nomenclature for the letterforms of roman type '' . Visible Language . 10 ( 1 ) : 41 -- 51 .
External links ( edit ) Wikimedia Commons has media related to Examples of typography terms . Armannsson , Sigurdur ( 6 July 2009 ) . `` Wallpaper : Font
Anatomy '' . Font.is . Full - size image ( 1920 × 1200 ) Dean , Paul ( 2008 ) . eXtreme Type Terminology -- Part 1 , Part 2 , Part 3 , Part 4 and Part 5 Devroye
, Luc . `` Hoefler & Frere - Jones '' . Retrieved 21 January 2015 . Frere - Jones , Tobias . `` Typeface Mechanics : 001 '' . Frere - Jones Type .
</item 7>

<item 8>
Typeface anatomy
This made it possible to mix typefaces from completely different genres such as sans - serifs and serifs and have the cap height , baseline and linespacing
match perfectly , something not possible with most digital fonts . It even allowed mixing of different sizes of type with a consistent baseline . It however had
the disadvantage of often forcing typefaces to be issued with cropped descenders compared to historical typefaces , to allow tight linespacing . A `` script
line '' or `` art line '' was used for more delicate fonts with long descenders . Titling capitals , meanwhile , were issued taking up the whole space of the
metal type area , with no room for descenders . See also ( edit ) Type design Type foundry Typography Typesetting References ( edit ) Jump up ^ Studer , Anton (
29 February 2016 ) . `` Is What I See What I Get ? -- Math & Optics in Type Design '' . Typographica . Retrieved 17 April 2016 . ^ Jump up to : Carter , Bob ;
Day , Ben ; Meggs , Philip ( 2007 ) . Typographic Design : Form and Communication .
</item 8>

<item 9>
Typeface anatomy
The close counter in e is also named eye . Angles of white space , as in w , are corners ( w has three corners ) ; the term is not used for angles of strokes .
The small corner formed by a serif , whether curved or angular , is called the serif bracket . Proportions ( edit ) A subtle change in proportion impacts weight
, perception , measure , and legibility . The letterform height compared to its stroke width modifies the aspect ; this slight change in weight sometimes helps
to create emphasis . The disparity between thick and thin strokes , known as stress , alters optical perception . As an example , the initial san serif
typefaces used a constantly thin stroke . The technological advances allow drawing thinner strokes . Condense type occupies less space than expanded type , then
the text written on a page can be reduced to half a page. The capline and x-height ratio improve or decrease word legibility . Metal type era ( edit ) During
the late metal type period , many fonts ( particularly in American typefounding ) were issued to `` common line '' . This meant that they were made to
standardised proportions , so that fonts of different typefaces could be mixed with no difficulty .
</item 9>

<item 10>
Typeface anatomy
Typographers also speak of an instroke , where one starts writing the letter , as at the top of acf , and an outstroke , where the pen leaves off , as at the
bottom of cejkty . Typefaces are born from the struggle between rules and results . Squeezing a square about 1 % helps it look more like a square ; to appear
the same height as a square , a circle must be measurably taller . The two strokes in an X are n't the same thickness , nor are their parallel edges actually
parallel ; the vertical stems of a lowercase alphabet are thinner than those of its capitals ; the ascender on ad is n't the same length as the descender on ap
, and so on . For the rational mind , type design can be a maddening game of drawing things differently in order to make them appear the same . Hoefler & Frere
- Jones A main vertical stroke is called a stem . The letter m has three , the left , middle , and right stems . The central stroke of an s is called the spine
.
</item 10>
=== END ITEM id=276a19b86d540f21 ===

=== ITEM id=f9aaa8d51f9c9d97 ===
Gold sentences and facts:
Sentence 0: Spike , occasionally referred to as Butch or Killer , is a stern but occasionally dumb American bulldog who is particularly disapproving of cats , but a softie when it comes to mice ( though in his debut appearance , Dog Trouble , Spike goes after both Tom and Jerry ) , and later , his son Tyke . In the shorts Jerry would often try to get Tom in trouble with Spike making him a shoo - in for a beating from the bulldog . Spike has a few weaknesses that Tom tries to capitalize upon : his possessiveness about his bone and his ticklishness . He made his first appearance in the 1942 Tom and Jerry cartoon Dog Trouble , and his first speaking role was in 1944 's The Bodyguard , where he was voiced by Billy Bletcher up until 1949 , from which point he was voiced by Daws Butler . Unlike his father Spike , Tyke does not speak . He only communicates by yapping , whimpering , growling , facial expressions and wagging his tail . In Tom and Jerry Kids , Tyke does have a speaking role in the program and is the first time that viewers were able to hear Tyke speak . Spike is very protective towards his son and gets very angry at Tom if Tyke is bothered or harmed . Although Tyke has spoken in Tom and Jerry Kids , he has laughed in one Tom and Jerry short . After Daws Butler , Maurice LaMarche , Frank Welker , John DiMaggio , Michael Donovan , Phil LaMarr and currently Rick Zieff would all perform Spike 's voice . Tyke 's vocal effects are provided by Frank Welker and speaking roles by Patric Zimmerman .
  Fact 0.0: Spike is occasionally referred to as Butch or Killer.
  Fact 0.1: Spike is a stern but occasionally dumb American bulldog.
  Fact 0.2: Spike is particularly disapproving of cats.
  Fact 0.3: Spike is a softie when it comes to mice.
  Fact 0.4: In his debut appearance, Dog Trouble, Spike goes after both Tom and Jerry.
  Fact 0.5: Spike later appears with his son, Tyke.
  Fact 0.6: In the shorts, Jerry would often try to get Tom in trouble with Spike.
  Fact 0.7: Jerry's trouble-making makes Tom a shoo-in for a beating from the bulldog Spike.
  Fact 0.8: Spike has a few weaknesses.
  Fact 0.9: Tom tries to capitalize on Spike's weaknesses.
  Fact 0.10: Spike's weaknesses include his possessiveness about his bone.
  Fact 0.11: Spike's weaknesses include his ticklishness.
  Fact 0.12: Spike made his first appearance in the 1942 Tom and Jerry cartoon Dog Trouble.
  Fact 0.13: Spike's first speaking role was in 1944's The Bodyguard.
  Fact 0.14: Billy Bletcher voiced Spike up until 1949.
  Fact 0.15: From 1949 onward, Spike was voiced by Daws Butler.
  Fact 0.16: Tyke does not speak, unlike his father Spike.
  Fact 0.17: Tyke only communicates by yapping.
  Fact 0.18: Tyke communicates by whimpering.
  Fact 0.19: Tyke communicates by growling.
  Fact 0.20: Tyke communicates by facial expressions.
  Fact 0.21: Tyke communicates by wagging his tail.
  Fact 0.22: Tyke has a speaking role in Tom and Jerry Kids.
  Fact 0.23: Tom and Jerry Kids was the first time viewers were able to hear Tyke speak.
  Fact 0.24: Spike is very protective towards his son Tyke.
  Fact 0.25: Spike gets very angry at Tom if Tyke is bothered or harmed.
  Fact 0.26: Tyke has spoken in Tom and Jerry Kids.
  Fact 0.27: Tyke has laughed in one Tom and Jerry short.
  Fact 0.28: After Daws Butler, Maurice LaMarche performed Spike's voice.
  Fact 0.29: After Daws Butler, Frank Welker performed Spike's voice.
  Fact 0.30: After Daws Butler, John DiMaggio performed Spike's voice.
  Fact 0.31: After Daws Butler, Michael Donovan performed Spike's voice.
  Fact 0.32: After Daws Butler, Phil LaMarr performed Spike's voice.
  Fact 0.33: Rick Zieff currently performs Spike's voice.
  Fact 0.34: Frank Welker provides Tyke's vocal effects.
  Fact 0.35: Patric Zimmerman provides Tyke's speaking roles.

Retrieved items:
<item 1>
List of Tom and Jerry characters
`` Tom '' is a generic name for a male cat . He is usually but not always , portrayed as living a comfortable , or even pampered life , while Jerry ( named ``
Jinx '' in his debut appearance ) is a small , brown , house mouse who always lives in close proximity to Tom . Despite being very energetic , determined and
much larger , Tom is no match for Jerry 's wits . Jerry also possesses surprising strength for his size , approximately the equivalent of Tom 's , lifting items
such as anvils with relative ease and withstanding considerable impacts . Although cats typically chase mice to consume them , it is quite rare for Tom to
actually try to consume Jerry . Most of his attempts are just to torment or humiliate Jerry , sometimes in revenge , and sometimes to obtain a reward from a
human for catching Jerry . By the final `` fade - out '' of each cartoon , Jerry usually emerges triumphant , while Tom is shown as the loser . However , other
results may be reached . On rare occasions , Tom triumphs , usually when Jerry becomes the aggressor or when he pushes Tom a little too far .
</item 1>

<item 2>
List of Tom and Jerry characters
Like many other early human characters , her face was never shown . Chuck Jones era ( edit ) The following characters were introduced in the theatrical shorts
that were produced by Chuck Jones in the 1960s . Tiny bulldog ( edit ) This nameless mouse - sized bulldog , designed similarly to Spike the Bulldog , is Jerry
's pet dog companion in only two shorts : The Cat 's Me - Ouch and Purr - Chance to Dream . His role is similar to Spike in cartoons such as The Bodyguard , Fit
To Be Tied , and Much Ado About Mousing , which is to defend Jerry against Tom . He is presumably the oldest son of Spike before Tyke , and this could be
identified by his looks and personality that is similar to his father Spike . Unlike Spike and Tyke he has a big bite ( similar to those of Looney Tunes '
Tasmanian Devil ) . Porpoise ( edit ) A shark who tries to eat Tom and Jerry . He appears in Cannery Rodent , Puss ' n ' Boats , Filet Meow and Surf - Bored Cat
.
</item 2>

<item 3>
List of Tom and Jerry characters
Trojan Dog ' in Tom and Jerry Comedy Show ) and movies thereafter . In his first appearance , the mad scientist is the owner of Jerry ( who is also his
assistant ) and has numerous cats trapped in a prison . When Jerry picks an orange cat named Lightning , the other cats flee in terror as the scientist pulls
him out for his latest experiment . He switches the mind of the orange cat with that of a bulldog and presents him as his present to Jerry . Thin lady ( edit )
An unnamed thin white lady became Tom 's owner in Buddies Thicker Than Water . Gene Deitch opted not to use Mammy 's character in his shorts , as he felt a ``
stereotypical black housekeep '' character `` did n't work in a modern context . '' Chuck Jones would also use her in The Unshrinkable Jerry Mouse and in the
re-animated version of Saturday Evening Puss . She also appeared in the movie Tom and Jerry : The Fast and the Furry . Like many other early human characters ,
her face was never shown .
</item 3>

<item 4>
List of Tom and Jerry characters
The most frequent love interest of Tom 's is Toodles Galore , who never has any dialogue in the cartoons . Despite five shorts ending with a depiction of Tom 's
apparent death , his demise is never permanent ; he even reads about his own death in a flashback in Jerry 's Diary . He appears to die in explosions in Mouse
Trouble ( after which he is seen in heaven ) , Yankee Doodle Mouse and in Safety Second , while in The Two Mouseketeers he is guillotined offscreen . Hanna -
Barbera era ( edit ) The following characters were introduced in the theatrical shorts that were directed by William Hanna and Joseph Barbera . Spike and Tyke (
edit ) Main article : Spike and Tyke ( characters ) Spike , occasionally referred to as Butch or Killer , is a stern but occasionally dumb American bulldog who
is particularly disapproving of cats , but a softie when it comes to mice ( though in his debut appearance , Dog Trouble , Spike goes after both Tom and Jerry )
, and later , his son Tyke .
</item 4>

<item 5>
List of Tom and Jerry characters
Butch is usually decipted as a homeless alley / street cat although in the 1956 short , Blue Cat Blues , Butch is apparently a millionaire who wins the
attention of Tom 's love interest due to his immense wealth much to Tom 's dismay . Butch also appears in the Spike and Tyke cartoon Scat Cats as a house cat
owned by George and Joan who in the original series own Tom . Toodles Galore ( edit ) Toodles Galore is an attractive white female cat , and is supposedly Tom
's usual love interest , although Tom is a reputed playboy , and had other love interests before and after Toodles . Toodles is the only love interest who
appeared more than twice , and is probably the most favored . During the classic era , Tom had to compete twice against Butch and even once against Spike for
Toodles 's affection , and he lost them all . In Casanova Cat , Toodles fell in love with Jerry . However , in her final short Love Me , Love My Mouse , she
remained with Tom .
</item 5>

<item 6>
List of Tom and Jerry characters
References ( edit ) Jump up ^ Deitch , Gene ( 2015 ) . Tom and Jerry ... and Gene in Tom and Jerry : The Gene Deitch Collection ( DVD ) . 20th Century Fox . Tom
and Jerry Characters Tom Cat Jerry Mouse Butch Nibbles Spike and Tyke Mammy Two Shoes Theatrical releases Tom and Jerry ( shorts ) Spike and Tyke ( shorts ) The
Movie Cameos Anchors Aweigh Dangerous When Wet DVD releases of shorts Spotlight Collection Golden Collection Chuck Jones Collection Classic Collection Deluxe
Anniversary Collection Gene Deitch Collection Television series Show ( 1975 ) Comedy Show Kids Show episodes Tales Show ( 2014 ) episodes Television specials
Hanna - Barbera 's 50th : A Yabba Dabba Doo Celebration The Mansion Cat Santa 's Little Helpers Direct - to - video films The Magic Ring Blast Off to Mars The
Fast and the Furry Shiver Me Whiskers A Nutcracker Tale Meet Sherlock Holmes The Wizard of Oz Robin Hood and His Merry Mouse Giant Adventure The Lost Dragon Spy
Quest Back to Oz Willy Wonka and the Chocolate Factory Other formats Video games Soundtrack Related William Hanna Joseph Barbera Scott Bradley Hanna - Barbera
Founders William Hanna
</item 6>

<item 7>
List of Tom and Jerry characters
He is based on the character from Robin Hood tales . Yvonne jockalong ( edit ) Love interest to both Tom and Jerry , she appears in Sky Bunny and The Tennis
Menace . Sergeant farce ( edit ) A police officer who appears in See Dr. Jackal and Hide and in The Police Kitten where he is the superior officer to both Tom
and Jerry . Alley Cat Gang ( edit ) The Tom and Jerry Comedy Show ( edit ) The following characters were introduced in the television series The Tom and Jerry
Comedy Show . Droopy ( edit ) Main article : Droopy Dog A Basset Hound often with a miserable look on his face , originally introduced in the Tex Avery shorts
of the 1940s . He first made a poster cameo in Matinee Mouse . He also appeared in the Droopy and Dripple segments of Tom and Jerry Kids . He is sometimes an
adversary of Spike . He would also appear in Tom and Jerry Tales and would appear in the direct - to - video films .
</item 7>

<item 8>
List of Tom and Jerry characters
In Tom and Jerry : Spy Quest , they are seen working for Doctor Zin . They also make an appearance in Tom and Jerry : Willy Wonka and the Chocolate Factory .
They were named after the Warner Bros. 1943 short Tin Pan Alley Cats . Tin is voiced by Greg Ellis , Pan by Jess Harnell and Alley by Richard McGonagle . Butch
( dog ) ( edit ) Main article : Butch Dog ( Tex Avery ) Butch ( Originally known as Spike , until renamed as Butch to avoid confusion with Spike from the Tom
and Jerry cartoons ) is the name of another bulldog from Tex Avery 's shorts in the 1940s and 1950s . He was one of the main antagonists in Droopy shorts (
although he was replaced by Hanna - Barbera 's Spike in the 1980s ) and also had a series of his own . He appears in a number of movies e.g. Tom and Jerry and
the Wizard of Oz and Tom and Jerry : The Magic Ring . References ( edit ) Jump up ^ Deitch , Gene ( 2015 ) .
</item 8>

<item 9>
List of Tom and Jerry characters
This backfired and left Mechano with a black face , while crying in frustration . He sneaks back and promptly shoots Tom , who attempts to shoot Mechano back
for his insubordination . However , the shot came out on the wrong end and Tom faints leaving Mechano laughing himself silly . He also appeared in the Tom and
Jerry Tales episode `` Tin Cat of Tomorrow '' where he is named Verminator 7000 . Mechano ( called `` Robot Cat '' ) also appeared as a fighter in the video
game , War of the Whiskers , and was voiced by Dee Bradley Baker . The St. Bernard ( edit ) The St. Bernard is a dog who first appeared in Puttin ' on the Dog
and then again in The A-Tom - Inable Snowman . Unlike other dogs in the series , he is not hostile to cats and even attends Tom medically . Eagle ( edit ) Eagle
is an Eagle that tries to steal Jerry for food from Tom .
</item 9>

<item 10>
List of Tom and Jerry characters
( as a ticking time bomb set by Tom ) . Toots ( edit ) Toots is the name given to three different characters that have appeared infrequently . The first of
these is a fluffy beige cat who appears in Puss n ' Toots and The Mouse Comes to Dinner , the latter being her only speaking role . She also has a cameo in
Smitten Kitten . This version of Toots also appeared in some 1940s Tom and Jerry comics . Toots is also a different cat by the same name who appears in The Zoot
Cat and in the Tom and Jerry Tales episode `` Kitty Cat Blues . '' She has occasionally and erroneously been referred to as `` Sheikie , '' but this is actually
Tom 's nickname , as seen on a gift card in the cartoon . Finally Toots is the name of a mouse who serves as Jerry 's love interest in Blue Cat Blues . King of
France ( edit ) In some of the mouseketeer series of cartoons , Tom serves the king of France . This is an obvious reference to Louis XVI of France , who is
shown as an extremely short - tempered and cruel monarch .
</item 10>
=== END ITEM id=f9aaa8d51f9c9d97 ===

=== ITEM id=d8feb424bf7ff376 ===
Gold sentences and facts:
Sentence 0: The National Legal Aid & Defender Association ( NLADA ) is the oldest and largest national , nonprofit membership organization devoted to advocating equal justice for all Americans and was established in 1911 .
  Fact 0.0: The National Legal Aid & Defender Association (NLADA) is the oldest national membership organization.
  Fact 0.1: The National Legal Aid & Defender Association (NLADA) is the largest national membership organization.
  Fact 0.2: The National Legal Aid & Defender Association (NLADA) is nonprofit.
  Fact 0.3: The National Legal Aid & Defender Association (NLADA) is devoted to advocating equal justice for all Americans.
  Fact 0.4: The National Legal Aid & Defender Association (NLADA) was established in 1911.

Retrieved items:
<item 1>
National Legal Aid & Defender Association
Other municipalities followed suit , and in the first decades of the 20th century most major cities had opened legal aid societies . In 1911 , legal aid
societies joined together to form the National Alliance of Legal Aid Societies . Arthur von Briesen of the Legal Aid Society of New York was the first president
of the organization that became the National Legal Aid & Defender Association ( NLADA ) in 1949 . The concept of free legal assistance for the poor was promoted
by the publication of Reginald Heber Smith 's Justice and the Poor in 1919 . Smith challenged the legal profession to consider it an obligation to see that
access to justice was available to all , without regard to ability to pay . `` Without equal access to the law , '' he wrote , `` the system not only robs the
poor of their only protection , but places in the hands of their oppressors the most powerful and ruthless weapon ever invented . As a result of Smith 's book ,
the American Bar Association created the Special Committee on Legal Aid Work . By the middle of the 20th century , virtually every major metropolitan area had
some kind of legal aid program .
</item 1>

<item 2>
American Bar Association Model Rules of Professional Conduct
Another important factor is that the legal profession is regulated very differently in California than it is in the rest of the United States . Outside of
California , the profession is regulated solely by the judicial branch of state government . Within California , the State Bar of California operates as a state
government agency under the supervision of the California Supreme Court ; but the State Bar was created by the California Legislature , and the Legislature has
full power to regulate the legal profession , if it so wishes . In particular , the State Bar can not collect annual member dues unless and until the California
Legislature passes an annual State Bar Dues Bill , which the governor must sign into law . When Governor Pete Wilson vetoed the dues bill in 1997 , the State
Bar effectively shut down operations except for self - funded activities like the California Bar Exam . In 2010 , Governor Arnold Schwarzenegger vetoed the dues
bill but a crisis was averted on that occasion . But the most important reason the California Bar is reluctant to adopt the ABA Model Rules is that most members
believe the ABA Model Rules would not sufficiently protect the California public from harm .
</item 2>

<item 3>
American Bar Association Model Rules of Professional Conduct
American Bar Association Model Rules of professional conduct - wikipedia American Bar Association Model Rules of professional conduct Jump to : navigation ,
search Professional responsibility Duties to the client Confidentiality Avoiding conflict of interest Due diligence and Competence ( law ) Avoid commingling
Avoid self - dealing Effective assistance Avoid fee splitting Withdrawal from representation Duties to the court Disclosure of perjury Disclosure of adverse
authority Duties to the profession Limitations on legal advertising Report misconduct Sources of law ABA Model Rules Penalties for misconduct Disbarment
Judicial misconduct The ABA Model Rules of Professional Conduct , created by the American Bar Association ( ABA ) , are a set of rules that prescribe baseline
standards of legal ethics and professional responsibility for lawyers in the United States . They were promulgated by the ABA House of Delegates upon the
recommendation of the Kutak Commission in 1983 . The rules are merely recommendations , or models , ( hence the name `` Model Rules '' ) and are not themselves
binding . However , having a common set of Model Rules facilitates a common discourse on legal ethics , and simplifies professional responsibility training as
well as the day - to - day application of such rules .
</item 3>

<item 4>
American Bar Association Model Rules of Professional Conduct
While either of their secretaries could be a member of the State Bar , they most likely are not . Similarly , the Board of Trustees that oversee the business
operations of the State Bar of California under supervision of the California Supreme Court is legally required to be a mix of attorneys licensed to practice in
California and `` public members '' who may not be licensed attorneys . So , while American attorneys are regulated by public sector agencies instead of private
sector law societies , most of the people who develop the regulatory policies are practising attorneys and sitting judges , so there is a degree of self -
regulation but not nearly to the same degree that solicitors and barristers in England and Wales , and other Commonwealth jurisdictions are self - regulated by
private sector associations . See also ( edit ) American Bar Association Model Code of Professional Responsibility American Bar Association References ( edit )
Jump up ^ Press Release : New Attorney Rules of Professional Conduct Announced , Communications Office of the New York Courts , 17 December 2008 .
</item 4>

<item 5>
American Bar Association Model Rules of Professional Conduct
However , the Model Rules have been adopted , in whole or in part , and sometimes in variation , as the rules of professional conduct for attorneys in 49 states
, the District of Columbia , and four of the five inhabited U.S. territories . The rules of professional conduct or professional responsibility adopted in a
particular state are legally enforceable against the lawyers of that state as well as any lawyer practicing there on a pro hac vice basis . On December 17 ,
2008 , New York announced that it would finally abandon the old ABA Model Code of Professional Responsibility ( the last state to do so ) and adopt a heavily
modified version of the ABA Model Rules , effective April 1 , 2009 . New York 's version of the Model Rules was created by adjusting the standard Model Rules to
reflect indigenous New York rules that had been incorporated over the years into its version of the Model Code .
</item 5>

<item 6>
American Bar Association Model Rules of Professional Conduct
Jump up ^ http://www.mebaroverseers.org/attorney_regulation/maine_conduct_rules.html Jump up ^
http://www.maine.gov/tools/whatsnew/index.php?topic=mebar_overseers_bar_rules&id=88185&v=article Jump up ^ http://www.cnmilaw.org/pdf/court_rules/R07.pdf
Northern Mariana Islands Rules of Attorney Discipline and Procedure . ^ Jump up to : Rogers , Joan C. ( 2 October 2014 ) . `` California Justices Tell State Bar
to Redo Proposals for Updating Lawyer Conduct Rules '' . ABA / BNA Lawyers ' Manual on Professional Conduct . Bloomberg BNA . Retrieved 15 October 2014 . ^ Jump
up to : Zitrin , Richard ( 10 October 2014 ) . `` Viewpoint : State Supreme Court Resets Ethics Rewrite '' . The Recorder . ALM Media Properties , LLC .
Retrieved 15 October 2014 . Jump up ^ http://www.calbar.ca.gov/AboutUs/BoardofTrustees.aspx Board of Trustees , State Bar of California .
</item 6>

<item 7>
National Legal Aid & Defender Association
NLADA is the only national membership organization devoted exclusively to ensuring legal representation for individuals who can not afford an attorney . NLADA
serves the equal justice community in two major ways : providing products and services and as a national voice in public policy and legislative debates on the
many issues affecting the equal justice community . NLADA also serves as a resource for those seeking more information on equal justice in the United States .
In recent years , the NLADA has honored individuals and organizations with a variety of awards at its annual conferences . Awards range from the Mary Ellen
Hamilton award , which recognizes a client of a legal aid or defender organization , to the Clara Shortridge Foltz Award , which honors organizations that have
pioneered important innovations in the field . Notable recipients of NLADA awards include former Attorney General Janet Reno , Bronx Defenders attorney David
Feige , and Hillary Clinton . Paul D. Borman was Chief Federal Defender , Legal Aid & Defender Association of Detroit , from 1979 to 1994 . He was the first
federal defender to be appointed to the federal bench , in 1994 .
</item 7>

<item 8>
American Bar Association Model Rules of Professional Conduct
In contrast to most Commonwealth countries , where transactional lawyers and litigators are separately licensed as `` solicitors '' and `` barristers , ''
respectively ; all American attorneys are licensed to practice as both solicitors and barristers ( therefore these terms are almost never used ) . Likewise ,
whereas solicitors in England and Wales are professionally self - regulated by the private - sector Law Society of England and Wales while barristers are self -
regulated by the private - sector General Council of the Bar , attorneys in all states except California , all territories and the District of Columbia are
generally regulated by the public sector judicial branch of state , district or territorial government , as officers of the court . The legal profession in
California is regulated jointly by the California Supreme Court , mostly through its agency the State Bar of California and by the California Legislature ,
which exercises legislative control over all professions . Within the public sector regulatory agencies , boards and organisations , officers are invariably
licensed members of the pertinent profession while key managerial and general staff typically are not . The Executive Director and Deputy Executive Director of
the State Bar of California , for example , are active members of the State Bar .
</item 8>

<item 9>
National Legal Aid & Defender Association
National legal aid & Defender Association - wikipedia National legal aid & Defender Association Jump to : navigation , search The National Legal Aid & Defender
Association ( NLADA ) is the oldest and largest national , nonprofit membership organization devoted to advocating equal justice for all Americans and was
established in 1911 . Contents ( hide ) 1 History 2 See also 3 References 4 External links History ( edit ) The Fourteenth Amendment to the United States
Constitution and its Equal Protection Clause provides Equal justice under law . Beginning in the late 1800s and throughout the early years of the 20th century ,
the American legal profession expressed its commitment to the concept of free legal assistance for poor people in the form of legal aid societies and bar
association legal aid committees . The first legal aid society , The German Society of New York , was founded in 1876 to protect German immigrants from
exploitation . Subsequently , the agency 's protection was extended to others and in 1890 it became the Legal Aid Society of New York . In 1888 , the Ethical
Culture Society of Chicago established by the Bureau of Justice was the first agency to offer legal assistance to individuals regardless of nationality , race
or gender .
</item 9>

<item 10>
American Bar Association Model Rules of Professional Conduct
In 2001 , the Commission for the Revision of the Rules of Professional Conduct of the State Bar of California began a comprehensive revision of the California
rules that would , among other things , convert them into a localized version of the ABA Model Rules . However , the Commission 's progress had been extremely
slow , simply because there are so many substantive and structural differences between the California rules and the ABA models . In October 2010 , the State Bar
of California formally adopted the Commission 's report , recommending adoption of 67 proposed changes to harmonise the California Rules of Professional Conduct
with the ABA Model Rules of Professional Responsibility . On September 19 , 2014 , however , for reasons that were not fully explained , the Supreme Court of
California rejected all of the proposed revisions and returned them to the State Bar . The Court 's letter directed the State Bar to start the process all over
again with a new Commission , and to submit a new set of revised rules by March 31 , 2017 . Professional regulation ( edit ) Except for the Patent Bar , there
is no national legal profession in the United States but there are 56 separate legal professions .
</item 10>
=== END ITEM id=d8feb424bf7ff376 ===
