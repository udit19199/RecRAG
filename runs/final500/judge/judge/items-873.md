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

This file has 5 items.

=== ITEM id=4020663acab27936 ===
Gold sentences and facts:
Sentence 0: `` Bad Boys '' is a 1987 song by the Jamaican reggae band Inner Circle , which obtained high popularity in the United States after its re-release in 1993 , peaking at number 8 on the Billboard Hot 100 and number 7 on the Top 40 Mainstream . It is well known as the opening theme to the U.S. TV show Cops .
  Fact 0.0: Bad Boys is a 1987 song.
  Fact 0.1: Bad Boys is a song by the Jamaican reggae band Inner Circle.
  Fact 0.2: Bad Boys obtained high popularity in the United States after its re-release in 1993.
  Fact 0.3: Bad Boys peaked at number 8 on the Billboard Hot 100.
  Fact 0.4: Bad Boys peaked at number 7 on the Top 40 Mainstream.
  Fact 0.5: Bad Boys is well known as the opening theme to the U.S. TV show Cops.

Retrieved items:
<item 1>
Jack McCoy
( hide ) Law & Order Characters Police Sgt . Max Greevey Sgt . Phil Cerreta Lennie Briscoe Joe Fontana Ed Green Cyrus Lupo Mike Logan Rey Curtis Nick Falco Nina
Cassady Kevin Bernard Captain Donald Cragen Lieutenant Anita Van Buren District attorneys Benjamin Stone Jack McCoy Michael Cutter Paul Robinette Claire Kincaid
Jamie Ross Abbie Carmichael Serena Southerlyn Alexandra Borgia Connie Rubirosa Adam Schiff Nora Lewin Arthur Branch Recurring characters Elizabeth Olivet , Ph.
D . Emil Skoda , M.D. Elizabeth Rodgers , M.D.
</item 1>

<item 2>
Bad Boys (Inner Circle song)
Bad Boys ( Inner Circle song ) - wikipedia Bad Boys ( Inner Circle song ) Jump to : navigation , search `` Bad Boys '' Swedish single picture sleeve , 1990
Single by Inner Circle from the album One Way and Bad Boys / Bad to the Bone B - side `` Sweat ( A La La La La Long ) '' Released 1991 , 1993 Format 7 - inch
single Genre Reggae Length 3 : 52 Label Island Songwriter ( s ) Ian Lewis Audio sample file help `` Bad Boys '' is a 1987 song by the Jamaican reggae band Inner
Circle , which obtained high popularity in the United States after its re-release in 1993 , peaking at number 8 on the Billboard Hot 100 and number 7 on the Top
40 Mainstream . It is well known as the opening theme to the U.S. TV show Cops .
</item 2>

<item 3>
Law & Order: Special Victims Unit
Commanding officers Donald Cragen Olivia Benson Assistant District Attorneys Alexandra Cabot Casey Novak Kim Greylek Rafael Barba Doctors George Huang Melinda
Warner Other characters Barry Moredock Brian Cassidy Elizabeth Donnelly Michael Cutter Sonya Paxton Jack McCoy Seasons Episodes Season 1 5 6 7 8 9 10 11 12 13
14 15 16 17 18 19 Notable episodes `` Payback '' `` 911 '' `` Authority '' `` Cold '' `` Zebras '' `` Behave '' `` Smoked '' `` Scorched Earth '' `` Justice
Denied '' `` Father Dearest '' `` Rhodium Nights '' `` Lost Reputation '' `` Above Suspicion '' `` Acceptable Loss '' `` Manhattan Vigil '' `` Comic Perversion
'' `` Glasgowman 's Wrath '' `` Chicago Crossover '' See also Cast members Awards and nominations Law & Order franchise Chicago franchise Chicago Fire
Characters Matthew Casey Kelly Severide Episodes Season 1 5 6 Chicago P.D.
</item 3>

<item 4>
Law & Order: Special Victims Unit
Law & Order : Special Victims Unit - Wikipedia Law & Order : Special Victims Unit Jump to : navigation , search For the current season , see Law & Order :
Special Victims Unit ( season 19 ) . Law & Order : Special Victims Unit Also known as Law & Order : SVU SVU Genre Police procedural Legal drama Thriller Created
by Dick Wolf Starring Christopher Meloni Mariska Hargitay Richard Belzer Dann Florek Michelle Hurd Stephanie March Ice - T B.D. Wong Diane Neal Tamara Tunie
Adam Beach Michaela McManus Danny Pino Kelli Giddish Raúl Esparza Peter Scanavino Opening theme Theme of Law & Order : Special Victims Unit Composer ( s ) Mike
Post Country of origin United States Original language ( s ) English No. of seasons 19 No. of episodes 421 ( list of episodes ) Production Executive producer (
s ) Dick Wolf Robert Palm ( season 1 ) David J.
</item 4>

<item 5>
Law & Order: Special Victims Unit (season 19)
Missing or empty title = ( help ) ( hide ) Law & Order : Special Victims Unit Characters Detectives Elliot Stabler Olivia Benson John Munch Monique Jeffries Fin
Tutuola Chester Lake Nick Amaro Amanda Rollins Dominick Carisi Jr . Commanding officers Donald Cragen Olivia Benson Assistant District Attorneys Alexandra Cabot
Casey Novak Kim Greylek Rafael Barba Doctors George Huang Melinda Warner Other characters Barry Moredock Brian Cassidy Elizabeth Donnelly Michael Cutter Sonya
Paxton Jack McCoy Seasons Episodes Season 1 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 Notable episodes `` Payback '' `` 911 '' `` Authority '' `` Cold '' ``
Zebras '' `` Behave '' `` Smoked '' `` Scorched Earth '' `` Justice Denied '' `` Father Dearest '' `` Rhodium Nights '' `` Lost Reputation '' `` Above Suspicion
'' `` Acceptable Loss '' `` Manhattan Vigil '' `` Comic Perversion '' `` Glasgowman 's Wrath '' `` Chicago Crossover '' See also Cast members Awards and
nominations Law & Order franchise Retrieved from `` https://en.
</item 5>

<item 6>
Blue Bloods (season 5)
Danny hears a waitress who sounds like one of the robbers . He later visits the woman 's husband and find he has no legs and is also suffering from traumatic
brain injury the result of combat . He tells the detectives he has had numerous delays in approval for treatment from the Veteran 's Administration , but that
his wife was able to come up with $100,000 in a `` fundraiser '' . Danny and Maria realize the bank robbery was the `` fundraiser '' , but they do not let on
the soldier . Danny also finds out the woman was cheating on her husband , he gets photographic proof and he uses it as leverage to get her to confess . Each
member of the group ultimately receives five years in prison , while Danny and Frank are able to pull some strings with the VA to get the soldier treated
immediately . Meanwhile , the work of a guerilla street artist named Spanky ( in the vein of Banksy ) is causing public scares in the form of pop - ups that
emerge from unattended bags . This creates a rift between Garrett and Gormley on how to treat these actions while maintaining the image of the NYPD .
</item 6>

<item 7>
Bad Boys (Inner Circle song)
Chart performance ( edit ) Weekly charts ( edit ) Year Chart Peak position 1991 Finland ( Suomen virallinen lista ) Norway ( VG - lista ) Sweden (
Sverigetopplistan ) Australia ( ARIA ) 25 Austria ( Ö3 Austria Top 40 ) 21 Canada Top Singles ( RPM ) 19 Germany ( Official German Charts ) 35 Ireland ( IRMA )
26 New Zealand ( Recorded Music NZ ) 5 UK Singles ( Official Charts Company ) 52 US Billboard Hot 100 8 US Billboard Hot Dance Music / Maxi - Singles Sales 15
US Billboard Hot R&B Singles 58 US Billboard Rhythmic Top 40 27 US Billboard Top 40 Mainstream 7 2006 US Billboard Hot Ringtones 10 2007 US Billboard Hot
Ringtones 20 Year - end charts ( edit ) Chart ( 1993 ) Rank New Zealand ( Recorded Music NZ ) 15 US Billboard Hot 100 69 In popular culture ( edit ) `` Bad Boys
'' was selected as the theme song to Cops because a field producer for the show happened to be a fan of Inner Circle .
</item 7>

<item 8>
Bad Boys (Inner Circle song)
203 . ISBN 1 - 894663 - 96 - 9 . External links ( edit ) Lyrics of this song at MetroLyrics ( hide ) Inner Circle Ian Lewis Roger Lewis Bernard `` Touter ''
Harvey Michael Sterling Lancelot Hall Trevor `` Skatta '' Bonnick Michael `` Ibo '' Cooper Stephen `` Cat '' Coore William Stewart William `` Bunny Rugs ''
Clarke Irvin `` Carrot '' Jarrett Milton `` Prilly '' Hamilton Charles Farquharson Calvin McKenzie Funky Brown Jacob Miller Lester Adderley Norman Grant Rick
Hunt Calton Coffie Dave Gonzales Kris Bentley Jr . Jazz Albums Bad to the Bone ( 1992 ) Singles `` Bad Boys '' `` Sweat ( A La La La La Long ) '' `` Games
People Play '' Related Discography Retrieved from `` https://en.wikipedia.org/w/index.
</item 8>

<item 9>
Blue Bloods (season 5)
Appearing regularly and receiving `` special guest star '' billing are Gregory Jbara as Deputy Commissioner of Public Information Garrett Moore , Robert
Clohessy as Lt. Sidney Gormley , and Abigail Hawk as Detective Baker , Frank 's primary aide . Main Cast ( edit ) Tom Selleck as Police Commissioner Francis ``
Frank '' Reagan Donnie Wahlberg as Detective Daniel `` Danny '' Reagan Bridget Moynahan as ADA Erin Reagan - Boyle Will Estes as Officer Jamison `` Jamie ''
Reagan Len Cariou as Henry Reagan Amy Carlson as Linda Reagan Sami Gayle as Nicole `` Nicky '' Reagan - Boyle Recurring Cast ( edit ) Marisa Ramirez as
Detective Maria Baez Vanessa Ray as Officer Edit `` Eddie '' Janko Abigail Hawk as Detective Abigail Baker Gregory Jbara as Deputy Commissioner of Public
Information Garrett Moore Robert Clohessy as Lieutenant Sidney Gormley Broadcast ( edit ) The season airs simultaneously on CTV in Canada . In Africa , it began
airing on Dstv on October 23 , 2014 . In the United Kingdom and Ireland , it began on Sky Atlantic on November 5 , 2014 .
</item 9>

<item 10>
Bad Boys (Inner Circle song)
It is well known as the opening theme to the U.S. TV show Cops . Contents ( hide ) 1 History 2 Chart performance 2.1 Weekly charts 2.2 Year - end charts 3 In
popular culture 4 See also 5 References 6 External links History ( edit ) The song was originally released in 1987 on the album One Way and obtained minor
airplay and poor sales . However , in 1992 , it was also included on the Bad to the Bone album , and the song was released as a single in 1993 , after the
unexpected success of their previous international hit single `` Sweat ( A La La La La Long ) '' . `` Bad Boys '' reached number 52 in the United Kingdom and
number 8 in the United States ; in the latter territory , it was certified Gold and sold 600,000 copies . The song was also released as B - side on Inner Circle
's 1992 single `` Sweat ( A La La La La Long ) '' . The song was also released as a single in mid-1991 in Norway ; it peaked at number one for three weeks .
</item 10>
=== END ITEM id=4020663acab27936 ===

=== ITEM id=2d7119d1228ec5e5 ===
Gold sentences and facts:
Sentence 0: A justice of the peace ( JP ) is a judicial officer , of a lower or puisne court , elected or appointed by means of a commission ( letters patent ) to keep the peace . In past centuries the term commissioner of the peace was often used with the same meaning . Depending on the jurisdiction , such justices dispense summary justice or merely deal with local administrative applications in common law jurisdictions . Justices of the peace are appointed or elected from the citizens of the jurisdiction in which they serve , and are ( or were ) usually not required to have any formal legal education in order to qualify for the office . Some jurisdictions have varying forms of training for JPs .
  Fact 0.0: A justice of the peace (JP) is a judicial officer.
  Fact 0.1: A justice of the peace is a judicial officer of a lower or puisne court.
  Fact 0.2: A justice of the peace is elected or appointed by means of a commission (letters patent).
  Fact 0.3: A justice of the peace is elected or appointed to keep the peace.
  Fact 0.4: In past centuries the term commissioner of the peace was often used with the same meaning as justice of the peace.
  Fact 0.5: Depending on the jurisdiction, justices of the peace dispense summary justice.
  Fact 0.6: Depending on the jurisdiction, justices of the peace merely deal with local administrative applications in common law jurisdictions.
  Fact 0.7: Justices of the peace are appointed or elected from the citizens of the jurisdiction in which they serve.
  Fact 0.8: Justices of the peace are (or were) usually not required to have any formal legal education in order to qualify for the office.
  Fact 0.9: Some jurisdictions have varying forms of training for justices of the peace.

Retrieved items:
<item 1>
Justice of the peace
Administrative justice of the peace are court officers appointed by the Minister of Justice , and perform duties such as receiving criminal informations and
issuing warrants . Presiding justice of the peace are appointed by commission under the Great Seal , and can try some criminal matters and issue warrants . They
are appointed from advocates of at least ten years ' standing and serve full - time until the age of 70 . Yukon ( edit ) In Yukon , justices of the peace are
lay officers of the court . They sit in the Justice of the Peace Court , which is part of the Territorial Court of Yukon . Hong Kong ( edit ) In Hong Kong , the
historical functions of justices of the peace have been replaced by full - time , legally qualified magistrates . Nowadays , justices of the peace are
essentially titles of honour given by the Government to community leaders , and to certain officials while they are in their terms of offices . They have no
judicial functions , and their main duties include visiting prisons , institutions for young offenders and drug addicts , psychiatric hospitals , remand homes ,
places of refuge , reception and detention centres , administering statutory declarations , and serving as members of advisory panels .
</item 1>

<item 2>
Justice of the peace
The court specifically recognized that in the aftermath of Gideon v. Wainwright ( 1963 ) , it made no sense to allow a case to be tried before a layperson
incapable of understanding the legal arguments of the attorney to whom the defendant was entitled under Gideon . The remaining justice courts ( as well as
municipal courts ) were eliminated by the passage of Proposition 220 in June 1998 , which merged all lower courts within the state judicial branch into the
superior courts ( the courts of general jurisdiction ) . Under current California law , all California judges must be licensed attorneys . Notably , the Supreme
Court of the United States sharply disagreed with California 's analysis of the Fourteenth Amendment in the landmark case of North v. Russell , 427 U.S. 328 (
1976 ) , in which the court held that Kentucky 's use of nonlawyer judges in its police courts was not a violation of the Fourteenth Amendment guarantees of due
process and equal protection of the laws .
</item 2>

<item 3>
Justice of the peace
United states ( edit ) In some US states , the justice of the peace is a judge of a court of limited jurisdiction , a magistrate , or a quasi-judicial official
with certain statutory or common law magisterial powers . Some states have special qualifications or unique features for the office . The justice of the peace
typically presides over a court that hears misdemeanor cases , traffic violations , and other petty criminal infractions . The justice of the peace may also
have authority over cases involving small debts , landlord and tenant disputes , or other small claims court proceedings . Proceedings before justices of the
peace are often faster and less formal than the proceedings in other courts . In some jurisdictions a party convicted or found liable before a justice of the
peace may have the right to a trial de novo before the judge of a higher court rather than an appeal strictly considered . A justice of the peace also performs
civil marriages . Arizona ( edit ) A justice of the peace has the same jurisdiction as a municipal magistrate with respect to traffic and misdemeanor cases and
restraining orders , though over cases whose affairs are not contained within the confines of a single municipality .
</item 3>

<item 4>
Justice of the peace
JPs , who are also volunteers , are selected through an extensive interview , written exam and practical testing . They are recommended by the state attorney -
general and appointed by the governor - in - council , and it is their job to authorise and witness statutory declarations and affidavits within the state of
Victoria . There are currently around 4,800 JPs serving in all areas of the state . Bail justices ( edit ) The role of a bail justice is to hear bail
applications ( under the Bail Act 1977 ) and to hear applications for interim accommodation orders for children ( under the Children and Young Persons Act 1989
) within Victoria . Bail justices , once appointed , may remain in their role until they turn 70 years of age ( although they must be under 65 at the time of
their appointment ) . They are often required to attend call outs and rule on bail applications or protection applications for children in danger on weekends
and late at night when the courts are closed , but they can also witness Victorian statutory declarations and affidavits . Candidates must successfully complete
a three - day training course run by the Department of Justice .
</item 4>

<item 5>
Justice of the peace
Until the Courts Act 2003 came into force , magistrates were tied to a particular area ( see magistrates ' courts committee , commission area , petty sessions
area ) . This has now been changed such that they are assigned to local justice areas , but less strongly . The Courts Act provides the current framework for
appointment of the justices , which is done by the Lord Chancellor in the name of sovereign . Justices can also be removed by the same mechanism . District
judges ( magistrates ' court ) -- previously known as stipendiary magistrates -- must have a seven years ' general legal qualification , and are appointed by
the sovereign on the advice of the Lord Chancellor . Before 1714 , magistrates were liable to be approached at any time and in any place by people legally
recognised as paupers , appealing for aid if parish authorities refused to provide any . It was relatively common for these magistrates to write out , on the
spot , an order requiring aid to be granted . Scotland ( edit ) See also : Judiciary of Scotland Within the Scottish legal system justices of the peace are lay
magistrates who currently sit in the justice of the peace courts .
</item 5>

<item 6>
Justice of the peace
A JP has contempt power of $100 and up to 3 days in jail per occurrence . A JP is also authorized to perform marriage ceremonies . Replacement with other courts
( edit ) In many states , the office of justice of the peace has been abolished or transferred to another court , such as the magistrate court . In larger
cities , cases may be heard in a municipal court which has jurisdiction only within that city . Most efforts to abolish the office of justice of the peace have
been led by the American Bar Association , which views non-lawyer judges as no longer necessary , as there are now far more persons with formal legal education
than in the past when justices of the peace were first used . California formerly had justice of the peace courts staffed by lay judges , but began phasing them
out after a landmark 1974 decision in which the Supreme Court of California unanimously held that it was a violation of federal due process ( under the
Fourteenth Amendment to the U.S. Constitution ) to allow a non-lawyer to preside over a criminal trial which could result in incarceration of the defendant .
</item 6>

<item 7>
Justice of the peace
Additionally , the Justice Court hears cases involving county ordinances ( ordinances enacted by the board of supervisors that apply only to unincorporated
areas ) , civil lawsuits up to a limit of $10,000 , small claims cases up to $2,500 , and issues evictions , called writs of restitution ( after a forcible
detainer or special detainer action ( eviction ) being successfully completed by a landlord ) . Justices of the peace , also called JPs , or Judges of the
Justice Court , are elected in partisan elections for four - year terms from specific districts called precincts . They have the same authority and
responsibility as all other judges in the state with respect to performing marriages , administrating oaths , adhering to the code of judicial conduct , and all
aspects of justice administration . However , Arizona law does not require justices of the peace to be lawyers . Many justices of the peace are not legally
trained , although all are required by the Arizona Supreme Court to complete a course at the Arizona Judicial College . As with JPs , municipal judges in
Arizona are not required to be lawyers . Arkansas ( edit ) This section does not cite any sources .
</item 7>

<item 8>
Justice of the peace
JPs are chosen under the Governor - General 's discretion . Malaysia ( edit ) In Malaysia , justices of the peace have largely been replaced in magistrates '
courts by legally qualified ( first - class ) stipendiary magistrates . However , state governments continue to appoint justices of the peace as honours . In
2004 , some associations of justices of the peace pressed the federal government to allow justices of the peace to sit as second - class magistrates in order to
reduce the backlog of cases in the courts . New Zealand ( edit ) A Justice of the Peace in New Zealand is someone of good stature in the community who is
authorized to witness and sign statutory declarations and affidavits as well as certify documents . They may also perform citizenship ceremonies and act as a
Visiting Justice in prisons the function of which is to record a prisoner 's grievance and decide on its validity . They have certain powers to issue search
warrants , and ( in conjunction with another justice of the peace ) may try minor criminal trials in the district court and exercise powers to remand defendants
in custody , grant bail , and adjourn court hearings .
</item 8>

<item 9>
Justice of the peace
Their judicial powers were replaced by full - time , legally qualified district justices ( now called district judges ) and their quasi-judicial powers by lay
Peace Commissioners . Peace commissioners may sign statutory declarations , and may issue summons and search warrants to the Garda Síochána ( Irish police ) . A
Peace Commissioner can not sign an Affidavit , in Ireland that function is performed by a Commissioner for Oaths . Jamaica ( edit ) A justice of the peace ( JP
) , according to the Ministry of Justice , is a person of unquestionable integrity who seeks to promote and protect the rights of the individual and helps to
provide justice to persons in a particular community . Additionally , the JP serves as a justice in petty court sessions , attends juvenile court sessions ,
issues summonses , considers applications for bail , explains and signs legal documents , sits on licensing panels , and gives counsel / advice . Any Jamaican
citizen that can speak and write English is eligible to become a JP . Any club / organisation / citizen can recommend someone to become JP for a community . JPs
are chosen under the Governor - General 's discretion .
</item 9>

<item 10>
Justice of the peace
Retrieved 9 December 2010 . Jump up ^ UK Parliament . District Courts ( Scotland ) Act 1975 as amended ( see also enacted form ) , from legislation.gov.uk .
Jump up ^ UK Parliament . Civic Government ( Scotland ) Act 1982 as amended ( see also enacted form ) , from legislation.gov.uk . Jump up ^ `` Justices of the
Peace - Judicial Office Holders - About the Judiciary - Judiciary of Scotland '' . www.scotland-judiciary.org.uk . Judicial Office for Scotland . 2017 .
Retrieved 3 April 2017 . Jump up ^ Scottish Parliament . Criminal Proceedings etc . ( Reform ) ( Scotland ) Act 2007 as amended ( see also enacted form ) , from
legislation.gov.uk. `` The Scottish Ministers may by order establish courts of summary criminal jurisdiction to be known as justice of the peace courts . ''
Jump up ^ `` What happens at a Scottish criminal court case - mygov. scot '' . www. mygov. scot . Scottish Government . 13 April 2017 . Retrieved 18 April 2017
.
</item 10>
=== END ITEM id=2d7119d1228ec5e5 ===

=== ITEM id=4d00f58936cef28a ===
Gold sentences and facts:
Sentence 0: From September 1947 , Toyota 's small - sized vehicles were sold under the name `` Toyopet '' ( トヨペット ) . The first vehicle sold under this name was the Toyopet SA , but it also included vehicles such as the Toyopet SB light truck , Toyopet Stout light truck , Toyopet Crown , Toyopet Master , and the Toyopet Corona . The word `` Toyopet ( Japanese article ) '' was a nickname given to the Toyota SA due to its small size , as the result of a naming contest the Toyota Company organized in 1947 . However , when Toyota eventually entered the American market in 1957 with the Crown , the name was not well received due to connotations of toys and pets . The name was soon dropped for the American market , but continued in other markets until the mid-1960s .
  Fact 0.0: From September 1947, Toyota's small-sized vehicles were sold under the name 'Toyopet' (トヨペット).
  Fact 0.1: The first vehicle sold under the name Toyopet was the Toyopet SA.
  Fact 0.2: Toyopet vehicles also included the Toyopet SB light truck.
  Fact 0.3: Toyopet vehicles also included the Toyopet Stout light truck.
  Fact 0.4: Toyopet vehicles also included the Toyopet Crown.
  Fact 0.5: Toyopet vehicles also included the Toyopet Master.
  Fact 0.6: Toyopet vehicles also included the Toyopet Corona.
  Fact 0.7: The word 'Toyopet' was a nickname given to the Toyota SA due to its small size.
  Fact 0.8: The Toyopet nickname resulted from a naming contest the Toyota Company organized in 1947.
  Fact 0.9: When Toyota entered the American market in 1957 with the Crown, the Toyopet name was not well received due to connotations of toys and pets.
  Fact 0.10: The Toyopet name was soon dropped for the American market.
  Fact 0.11: The Toyopet name continued in other markets until the mid-1960s.

Retrieved items:
<item 1>
Toyota
North America ( edit ) Main article : Toyota Motor Engineering & Manufacturing North America Toyota Motor North America headquarters is located in Plano , Texas
, and operates as a holding company in North America . Its manufacturing headquarters is located in Erlanger , Kentucky , and is known as Toyota Motor
Engineering & Manufacturing North America . Toyota Canada Inc. has been in production in Canada since 1983 with an aluminium wheel plant in Delta , British
Columbia , which currently employs a workforce of roughly 260 . Its first vehicle assembly plant , in Cambridge , Ontario , since 1988 , now produces Corolla
compact cars , Matrix crossover vehicles , and Lexus RX 350 luxury SUVs , with a workforce of 4,300 workers . Its second assembly operation in Woodstock ,
Ontario , began manufacturing the RAV4 late in 2008 . In 2006 , Toyota 's subsidiary Hino Motors opened a heavy duty truck plant , also in Woodstock , employing
45 people and producing 2000 trucks annually .
</item 1>

<item 2>
Toyota
Japan : Toyota Motor Corporation . January 29 , 2013 . Retrieved February 3 , 2014 . Jump up ^ `` December 2013 Sales Chart '' ( Press release ) . USA : Toyota
. January 2013 . Retrieved January 10 , 2014 . Jump up ^ `` TMC Announces Results for December 2014 and CY2014 '' . Jump up ^ `` Toyota U.S. Sales Summary '' (
Press release ) . USA : Toyota . Retrieved February 14 , 2015 . Jump up ^ `` TMC Announces Results for December 2015 and CY2015 '' . Jump up ^ `` TMC Announces
Results for December 2016 and CY2016 '' . Jump up ^ Toyota : A history of the First 50 Years . Toyota Motor Corporation . 1988 . ISBN 0 - 517 - 61777 - 3 . ,
p461 . ^ Jump up to : Harden , Blaine ( July 13 , 2008 ) . `` Japan 's Killer Work Ethic '' . Washington Post Foreign Service .
</item 2>

<item 3>
Toyota
The company was number one in global automobile sales for the first quarter of 2008 . In 2007 , Toyota released an update of its full - sized truck , the Tundra
, produced in two American factories , one in Texas and one in Indiana . Motor Trend named the Tundra `` Truck of the Year '' , and the 2007 Toyota Camry `` Car
of the Year '' for 2007 . It also began the construction of two new factories , one to build the RAV4 in Woodstock , Ontario , Canada , and the other to build
the Toyota Prius in Blue Springs , Mississippi , USA . This plant was originally intended to build the Toyota Highlander , but Toyota decided to use the plant
in Princeton , Indiana , USA , instead . The company has also found recent success with its smaller models -- the Corolla and Yaris . 2010s ( edit ) In 2011 ,
Toyota , along with large parts of the Japanese automotive industry , suffered from a series of natural disasters . The 2011 Tōhoku earthquake and tsunami led
to a severe disruption of the supplier base and a drop in production and exports .
</item 3>

<item 4>
Toyota
1990s ( edit ) In the 1990s , Toyota began to branch out from producing mostly compact cars by adding many larger and more luxurious vehicles to its lineup ,
including a full - sized pickup , the T100 ( and later the Tundra ) , several lines of SUVs , a sport version of the Camry , known as the Camry Solara , and the
Scion brand . They would also launch newer and arguably more iconic iterations of their sports cars , namely the MR2 , Celica , and Supra during this era , all
of which have already become icons of the 1990s . With a major presence in Europe , due to the success of Toyota Team Europe , the corporation decided to set up
Toyota Motor Europe Marketing and Engineering , TMME , to help market vehicles in the continent . Two years later , Toyota set up a base in the United Kingdom ,
TMUK , as the company 's cars had become very popular among British drivers . Bases in Indiana , Virginia , and Tianjin were also set up . In 1999 , the company
decided to list itself on the New York and London Stock Exchanges .
</item 4>

<item 5>
Toyota
However , when Toyota eventually entered the American market in 1957 with the Crown , the name was not well received due to connotations of toys and pets . The
name was soon dropped for the American market , but continued in other markets until the mid-1960s . 1960s -- 1970s ( edit ) By the early 1960s , the US had
begun placing stiff import tariffs on certain vehicles . The so - called `` chicken tax '' of 1964 placed a 25 % tax on imported light trucks . In response to
the tariff , Toyota , Nissan Motor Co. and Honda Motor Co. began building plants in the US by the early 1980s . 1980s ( edit ) With over 40 million sold , the
Corolla is one of the most popular and best selling cars in the world . Toyota received its first Japanese Quality Control Award at the start of the 1980s and
began participating in a wide variety of motorsports . Due to the 1973 oil crisis , consumers in the lucrative US market began turning to small cars with better
fuel economy .
</item 5>

<item 6>
Toyota
American car manufacturers had considered small economy cars to be an entry - level product , and their small vehicles employed a low level of quality to keep
the price low . Conservative Toyota held on to rear - wheel - drive designs for longer than most ; while a clear first in overall production they were only
third in production of front - wheel - drive cars in 1983 , behind Nissan and Honda . In part due to this , Nissan 's Sunny managed to squeeze by the Corolla in
numbers built that year . In 1982 , the Toyota Motor Company and Toyota Motor Sales merged into one company , the Toyota Motor Corporation . Two years later ,
Toyota entered into a joint venture with General Motors called the New United Motor Manufacturing , Inc , NUMMI , operating an automobile - manufacturing plant
in Fremont , California . The factory was an old General Motors plant that had been closed for two years . Toyota then started to establish new brands at the
end of the 1980s , with the launch of their luxury division Lexus in 1989 .
</item 6>

<item 7>
Toyota
Toyota - wikipedia Toyota Jump to : navigation , search For other uses , see Toyota ( disambiguation ) . Toyota Motor Corporation hkl headquarters in Toyota
City , Japan Type Public ( K.K. ) Traded as TYO : 7203 LSE : TYT NYSE : TM TOPIX Core 30 Component Industry Automotive Founded August 28 , 1937 ; 80 years ago (
1937 - 08 - 28 ) Headquarters Toyota , Aichi , Japan Area served Worldwide Key people Kiichiro Toyoda ( Chairman ) Shigeru Hayakawa ( Vice Chairman ) Takeshi
Uchiyamada ( President ) Products Automobiles , luxury vehicles , commercial vehicles , engines Production output 16,985,870 units ( CY 2015 ) Services Banking
, financing , leasing Revenue ¥ 28.40 trillion ( 2016 ) Operating income ¥ 2.85 trillion ( 2016 ) Profit ¥ 2.31 trillion ( 2016 ) Total assets ¥ 47.42 trillion
( 2016 ) Total equity ¥ 17.22 trillion ( 2016 ) Owner Toyota Industries ( 6.73 % ) Nippon Life ( 3.
</item 7>

<item 8>
Toyota
In Chinese - speaking countries using simplified Chinese characters ( e.g. China ) , Toyota is known as `` 丰田 '' ( pronounced as Fēngtián in Mandarin Chinese
and Hɔng Tshan in Minnanese ) . These are the same characters as the founding family 's name `` Toyoda '' in Japanese . Logo until 1989 , now still used as a
corporate logo Logo since 1989 Japan ( edit ) Toyota Store Aichi Toyopet Store , Saitama Corolla Store Kyotanabe Netz Store , Tsurumi - ku , Osaka Wikimedia
Commons has media related to Toyota dealerships . In Japan , Toyota currently maintains separate dealership sales channels . The first sales channel established
in 1946 called `` Toyota Store '' ( トヨタ 店 Toyota Mise ) sells large luxury sedans such as the Toyota Century , and the Toyota Crown . In 1955 , the `` Toyopet
Store '' ( トヨペット 店 Toyopetto - ten ) arrived , originally established to sell the Toyota Corona and the Toyopet ToyoAce truck .
</item 8>

<item 9>
Toyota
The principle of jidoka , which means the machine stops itself when a problem occurs , became later a part of the Toyota Production System . Looms were built on
a small production line . In 1929 , the patent for the automatic loom was sold to the British company Platt Brothers , generating the starting capital for the
automobile development . Mass production of Toyoda automated loom , displayed at the Toyota Museum in Nagakute - cho , Aichi - gun , Aichi Pref . Japan The
production of Toyota automobiles was started in 1933 as a division of Toyoda Automatic Loom Works devoted to the production of automobiles under the direction
of the founder 's son , Kiichiro Toyoda . Its first vehicles were the A1 passenger car and the G1 in 1935 . The Toyota Motor Co. was established as an
independent company in 1937 Toyoda Standard Sedan AA 1936 Vehicles were originally sold under the name `` Toyoda '' ( トヨダ ) , from the family name of the
company 's founder , Kiichirō Toyoda . In April 1936 , Toyoda 's first passenger car , the Model AA , was completed .
</item 9>

<item 10>
Toyota
Toyota Technical Center , Ann Arbor Charter Township , Michigan Toyota has a large presence in the United States with six major assembly plants in Huntsville ,
Alabama , Georgetown , Kentucky , Princeton , Indiana , San Antonio , Texas , Buffalo , West Virginia , and Blue Springs , Mississippi . Toyota had a joint -
venture operation with General Motors at New United Motor Manufacturing Inc. in Fremont , California , which began in 1984 and ended in 2009 . It still has a
joint venture with Subaru at Subaru of Indiana Automotive , Inc. in Lafayette , Indiana , which started in 2006 . In these assembly plants , the Camry and the
Tundra are manufactured , among others . Toyota marketing , sales , and distribution in the US are conducted through a separate subsidiary , Toyota Motor Sales
, U.S.A. , Inc . It has started producing larger trucks , such as the new Tundra , to go after the large truck market in the United States . Toyota is also
pushing hybrid vehicles in the US such as the Prius , Camry Hybrid , Highlander Hybrid , and various Lexus products . Currently , Toyota has no plans to offer
diesel motor options in its North American products , including the light - duty pickup trucks .
</item 10>
=== END ITEM id=4d00f58936cef28a ===

=== ITEM id=a2fa8806d87c5ebc ===
Gold sentences and facts:
Sentence 0: A vanishing point is a point on the image plane of a perspective drawing where the two - dimensional perspective projections ( or drawings ) of mutually parallel lines in three - dimensional space appear to converge . When the set of parallel lines is perpendicular to a picture plane , the construction is known as one - point perspective , and their vanishing point corresponds to the oculus , or `` eye point '' , from which the image should be viewed for correct perspective geometry . Traditional linear drawings use objects with one to three sets of parallels , defining one to three vanishing points .
  Fact 0.0: A vanishing point is a point on the image plane of a perspective drawing.
  Fact 0.1: In a perspective drawing, the two-dimensional perspective projections, or drawings, of mutually parallel lines in three-dimensional space appear to converge at a vanishing point.
  Fact 0.2: When the set of parallel lines is perpendicular to a picture plane, the construction is known as one-point perspective.
  Fact 0.3: In one-point perspective, the vanishing point corresponds to the oculus, or "eye point", from which the image should be viewed for correct perspective geometry.
  Fact 0.4: Traditional linear drawings use objects with one to three sets of parallels.
  Fact 0.5: Traditional linear drawings define one to three vanishing points.

Retrieved items:
<item 1>
Vanishing point
Vanishing point - wikipedia Vanishing point Jump to : navigation , search For other uses , see Vanishing point ( disambiguation ) . A vanishing point can be
seen at the far end of this railroad . A vanishing point is a point on the image plane of a perspective drawing where the two - dimensional perspective
projections ( or drawings ) of mutually parallel lines in three - dimensional space appear to converge . When the set of parallel lines is perpendicular to a
picture plane , the construction is known as one - point perspective , and their vanishing point corresponds to the oculus , or `` eye point '' , from which the
image should be viewed for correct perspective geometry . Traditional linear drawings use objects with one to three sets of parallels , defining one to three
vanishing points . Part of a series on Graphical projection Planar ( show ) Parallel projection Orthographic projection Multiview projection Axonometric
projection Isometric projection Oblique projection Perspective projection Curvilinear perspective Reverse perspective Views ( show ) 2.
</item 1>

<item 2>
Vanishing point
5 D Bird 's - eye view Cross section Cutaway drawing Exploded view drawing Fisheye lens Panorama Worm 's - eye view Zoom lens Topics ( show ) 3D projection
Anamorphosis Axonometry Computer graphics Computer - aided design Descriptive geometry Engineering drawing Map projection Picture plane Plans ( drawings )
Projection ( linear algebra ) Projection plane Projective geometry Stereoscopy Technical drawing Vanishing point Video game graphics Viewing frustum Contents (
hide ) 1 Vector notation 2 Theorem 3 Vanishing line 4 Properties of vanishing points 5 Curvilinear and reverse perspective 6 Detection of vanishing points 7
Applications of vanishing points 8 See also 9 References 10 External links Vector notation ( edit ) A 2D construction of perspective viewing , showing the
formation of a vanishing point The vanishing point may also be referred to as the `` direction point '' , as lines having the same directional vector , say D ,
will have the same vanishing point or converge at the same vanishing points .
</item 2>

<item 3>
Vanishing point
This is one - point perspective . Similarly , when the image plane intersects two world - coordinate axes , lines parallel to those planes will meet at infinity
and form two vanishing points . This is called two - point perspective . In three - point perspective the image plane intersects the x , y , and z axes and
therefore lines parallel to these axes intersect , resulting in three different vanishing points . See also : Property 1 below Theorem ( edit ) The vanishing
point theorem is the principal theorem in the science of perspective . It says that the image in a picture plane π of a line L in space , not parallel to the
picture , is determined by its intersection with π and its vanishing point . Some authors have used the phrase , `` the image of a line includes its vanishing
point '' . Guidobaldo del Monte gave several verifications , and Humphry Ditton called the result the `` main and Great Proposition '' . Brook Taylor wrote the
first book in English on perspective in 1714 , which introduced the term `` vanishing point '' and was the first to fully explain the geometry of multipoint
perspective , and historian Kirsti Andersen compiled these observations .
</item 3>

<item 4>
Drawing
Typically this convergence point is somewhere along the horizon , as buildings are built level with the flat surface . When multiple structures are aligned with
each other , such as buildings along a street , the horizontal tops and bottoms of the structures typically converge at a vanishing point . When both the fronts
and sides of a building are drawn , then the parallel lines forming a side converge at a second point along the horizon ( which may be off the drawing paper . )
This is a two - point perspective . Converging the vertical lines to a third point above or below the horizon then produces a three - point perspective . Depth
can also be portrayed by several techniques in addition to the perspective approach above . Objects of similar size should appear ever smaller the further they
are from the viewer . Thus the back wheel of a cart appears slightly smaller than the front wheel . Depth can be portrayed through the use of texture . As the
texture of an object gets further away it becomes more compressed and busy , taking on an entirely different character than if it was close . Depth can also be
portrayed by reducing the contrast in more distant objects , and by making their colors less saturated .
</item 4>

<item 5>
Vanishing point
She notes , in terms of projective geometry , the vanishing point is the image of the point at infinity associated with L , as the sightline from O through the
vanishing point is parallel to L . Vanishing line ( edit ) As a vanishing point originates in a line , so a vanishing line originates in a plane α that is not
parallel to the picture π . Given the eye point O , and β the plane parallel to α and lying on O , then the vanishing line of α is β ∩ π . For example , when α
is the ground plane and β is the horizon plane , then the vanishing line of α is the horizon line β ∩ π . Anderson notes , `` Only one particular vanishing line
occurs , often referred to as the `` horizon '' . To put it simply , the vanishing line of some plane , say α , is obtained by the intersection of the image
plane with another plane , say β , parallel to the plane of interest ( α ) , passing through the camera center . For different sets of lines parallel to this
plane α , their respective vanishing points will lie on this vanishing line .
</item 5>

<item 6>
Vanishing point
say D , will have the same vanishing point or converge at the same vanishing points . Mathematically , let q ≡ ( x , y , f ) be a point lying on the image plane
, where f is the focal length ( of the camera associated with the image ) , and let v ≡ ( x / h , y / h , f / h ) be the unit vector associated with q , where h
= √ x + y + f . If we consider a straight line in space S with the unit vector n ≡ ( n , n , n ) and its vanishing point v , the unit vector associated with v
is equal to n , assuming both are assumed to point towards the image plane . When the image plane is parallel to two world - coordinate axes , lines parallel to
the axis which is cut by this image plane will meet at infinity i.e. at the vanishing point . Lines parallel to the other two axes will not form vanishing
points as they are parallel to the image plane . This is one - point perspective .
</item 6>

<item 7>
Vanishing point
With the assumption that the environment consists of objects with only parallel or perpendicular sides , also called Lego - land , using vanishing points
constructed in a single image of the scene they recovered the 3D geometry of the scene . Similar ideas are also used in the field of robotics , mainly in
navigation and autonomous vehicles , and in areas concerned with object detection . See also ( edit ) Graphical projection References ( edit ) ^ Jump up to :
Kirsti Andersen ( 2007 ) Geometry of an Art , p. xxx , Springer , ISBN 0 - 387 - 25961 - 9 ^ Jump up to : B. Caprile , V. Torre ( 1 ) `` Using Vanishing Points
for Camera Calibration '' , International Journal of Computer Vision , Volume 4 , Issue 2 , pp. 127 - 139 , March 1990 Jump up ^ H. Ditton ( 1712 ) Treatise on
Perspective , p 45 Jump up ^ S.T. Barnard ' Interpreting Perspective Images `` , Artificial Intelligence 21 , 1983 , pp.
</item 7>

<item 8>
Vanishing point
The horizon line is a theoretical line that represents the eye level of the observer . If the object is below the horizon line , its vanishing lines angle up to
the horizon line . If the object is above , they slope down . All vanishing lines end at the horizon line . Properties of vanishing points ( edit ) 1 .
Projections of two sets of parallel lines lying in some plane π appear to converge , i.e. the vanishing point associated with that pair , on a horizon line , or
vanishing line H formed by the intersection of the image plane with the plane parallel to π and passing through the pinhole . Proof : Consider the ground plane
π , as y = c which is , for the sake of simplicity , orthogonal to the image plane . Also , consider a line L that lies in the plane π , which is defined by the
equation ax + bz = d .
</item 8>

<item 9>
Vanishing point
Using perspective pinhole projections , a point on L projected on the image plane will have coordinates defined as , x ′ = f x / z = f d − bz / az y ′ = f y / z
= f c / z This is the parametric representation of the image L ′ of the line L with z as the parameter . When z → − ∞ it stops at the point ( x ′ , y ′ ) = ( −
fb / a , 0 ) on the x ′ axis of the image plane . This is the vanishing point corresponding to all parallel lines with slope − b / a in the plane π . All
vanishing points associated with different lines with different slopes belonging to plane π will lie on the x ′ axis , which in this case is the horizon line .
2 . Let A , B , and C be three mutually orthogonal straight lines in space and v ≡ ( x , y , f ) , v ≡ ( x , y , f ) , v ≡ ( x , y , f ) be the three
corresponding vanishing points respectively .
</item 9>

<item 10>
Vanishing point
If we know the coordinates of one of these points , say v , and the direction of a straight line on the image plane , which passes through a second point , say
v , we can compute the coordinates of both v and v 3 . Let A , B , and C be three mutually orthogonal straight lines in space and v ≡ ( x , y , f ) , v ≡ ( x ,
y , f ) , v ≡ ( x , y , f ) be the three corresponding vanishing points respectively . The orthocenter of the triangle with vertices in the three vanishing
points is the intersection of the optical axis and the image plane . Curvilinear and reverse perspective ( edit ) A curvilinear perspective is a drawing with
either 4 or 5 vanishing points . In 5 - point perspective the vanishing points are mapped into a circle with 4 vanishing points at the cardinal headings N , W ,
S , E and one at the circle 's origin . A reverse perspective is a drawing with vanishing points that are placed outside the painting with the illusion that
they are `` in front of '' the painting . Single point perspective projection . Double point perspective projection .
</item 10>
=== END ITEM id=a2fa8806d87c5ebc ===

=== ITEM id=851c658c3c421e91 ===
Gold sentences and facts:
Sentence 0: The provisions relating to the right to property were changed a number of times . The Forty - Fourth Amendment of 1978 deleted the right to property from the list of fundamental rights A new provision , Article 300 - A , was added to the constitution , which provided that `` no person shall be deprived of his property save by authority of law '' . Thus if a legislature makes a law depriving a person of his property , there would be no obligation on the part of the State to pay anything as compensation . The aggrieved person shall have no right to move the court under Article 32 . Thus , the right to property is no longer a fundamental right , though it is still a constitutional right . If the government appears to have acted unfairly , the action can be challenged in a court of law by aggrieved citizens .
  Fact 0.0: The provisions relating to the right to property were changed a number of times.
  Fact 0.1: The Forty-Fourth Amendment of 1978 deleted the right to property from the list of fundamental rights.
  Fact 0.2: A new provision, Article 300-A, was added to the constitution.
  Fact 0.3: Article 300-A provides that no person shall be deprived of his property save by authority of law.
  Fact 0.4: If a legislature makes a law depriving a person of his property, the State has no obligation to pay anything as compensation.
  Fact 0.5: An aggrieved person shall have no right to move the court under Article 32 in that case.
  Fact 0.6: The right to property is no longer a fundamental right.
  Fact 0.7: The right to property is still a constitutional right.
  Fact 0.8: If the government appears to have acted unfairly, the action can be challenged in a court of law by aggrieved citizens.

Retrieved items:
<item 1>
Fundamental rights in India
The right to constitutional remedies is present for enforcement of Fundamental Rights . The right to privacy is an intrinsic part of Article 21 that protects
life and liberty of the citizens . Fundamental rights for Indians have also been aimed at overturning the inequalities of pre-independence social practices .
Specifically , they have also been used to abolish untouchability and thus prohibit discrimination on the grounds of religion , race , caste , sex , or place of
birth . They also forbid trafficking of human beings and forced labour ( a crime ) . They also protect cultural and educational rights of religious and
linguistic minorities by allowing them to preserve their languages and also establish and administer their own education institutions . They are covered in Part
III ( Articles 12 to 35 ) of Indian constitution .
</item 1>

<item 2>
Fundamental rights in India
. The Times of India . Retrieved 8 July 2010 . Jump up ^ `` Court rejects plea to make property a fundamental right '' . The Hindu . Chennai , India . 19
October 2010 . Jump up ^ `` Education is now a fundamental right of every child '' . Hindustan Times . 1 April 2010 . Archived from the original on 9 July 2010
. Retrieved 8 July 2010 . Footnotes ( edit ) Links to related articles ( hide ) Government of India Constitution Constitution Fundamental Rights , Directive
Principles and Fundamental Duties Federalism ( Union List State List Concurrent List ) Executive Union Government Prime Minister ( list ) Cabinet Ministers
President ( list ) Vice President ( list ) Civil Services All India Services ( IAS IFS / IFoS IPS ) Cabinet Secretary Secretaries ( Defence Finance Foreign Home
) State governments Chief Minister ( list ) Chief Secretary Governors ( list ) Legislative Parliament Lok Sabha ( Speaker ) Rajya Sabha ( Chairman ) Vidhan
Sabha Vidhan Parishad Leader of the Opposition Judiciary Supreme Court ( Chief Justice ) High Courts District Courts Retrieved from ``
https://en.wikipedia.org/w/index.
</item 2>

<item 3>
Fundamental rights in India
The provisions relating to the right to property were changed a number of times . The Forty - Fourth Amendment of 1978 deleted the right to property from the
list of fundamental rights A new provision , Article 300 - A , was added to the constitution , which provided that `` no person shall be deprived of his
property save by authority of law '' . Thus if a legislature makes a law depriving a person of his property , there would be no obligation on the part of the
State to pay anything as compensation . The aggrieved person shall have no right to move the court under Article 32 . Thus , the right to property is no longer
a fundamental right , though it is still a constitutional right . If the government appears to have acted unfairly , the action can be challenged in a court of
law by aggrieved citizens . The liberalisation of the economy and the government 's initiative to set up special economic zones has led to many protests by
farmers and have led to calls for the reinstatement of the fundamental right to private property .
</item 3>

<item 4>
Fundamental rights in India
They are covered in Part III ( Articles 12 to 35 ) of Indian constitution . Contents ( hide ) 1 Genesis 2 Significance and characteristics 3 Right to equality 4
Right to freedom 5 Right against exploitation 6 Right to freedom of religion 7 Right to life 8 Cultural and Educational rights 9 Right to constitutional
remedies 10 Right to Privacy 11 Critical analysis 12 Amendments 13 Fundamental rights not sacrosanct 13.1 Validity of Article 31B 13.2 Amendment to Article 31C
13.3 Right to property 13.4 Right to education 14 See also 15 References 16 Footnotes Genesis ( edit ) The development of constitutionally guaranteed
fundamental human rights in India was inspired by historical examples such as England 's Bill of Rights ( 1689 ) , the United States Bill of Rights ( approved
on 17 September 1787 , final ratification on 15 December 1791 ) and France 's Declaration of the Rights of Man ( created during the revolution of 1789 , and
ratified on 26 August 1789 ) .
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
These fundamental rights help not only in protection but also the prevention of gross violations of human rights . They emphasise on the fundamental unity of
India by guaranteeing to all citizens the access and use of the same facilities , irrespective of background . Some fundamental rights apply for persons of any
nationality whereas others are available only to the citizens of India . The right to life and personal liberty is available to all people and so is the right
to freedom of religion . On the other hand , freedoms of speech and expression and freedom to reside and settle in any part of the country are reserved to
citizens alone , including non-resident Indian citizens . The right to equality in matters of public employment can not be conferred to overseas citizens of
India . Fundamental rights primarily protect individuals from any arbitrary state actions , but some rights are enforceable against individuals . For instance ,
the Constitution abolishes untouchability and also prohibits begar . These provisions act as a check both on state action as well as the action of private
individuals . However , these rights are not absolute or uncontrolled and are subject to reasonable restrictions as necessary for the protection of general
welfare . They can also be selectively curtailed .
</item 6>

<item 7>
Fundamental rights in India
It is the principal foundation of all other rights and liberties , and guarantees the following : Equality before law : Article 14 of the constitution
guarantees that all people shall be equally protected by the laws of the country . It means that the State will treat people in the same circumstances alike .
This article also means that individuals , whether citizens of India or otherwise shall be treated differently if the circumstances are different . Social
equality and equal access to public areas : Article 15 of the constitution states that no person shall be discriminated on the basis of religion , race , caste
, sex or place of birth . Every person shall have equal access to public places like public parks , museums , wells , bathing ghats and temples etc . However ,
the State may make any special provision for women and children . Special provisions may be made for the advancements of any socially or educationally backward
class or scheduled castes or scheduled tribes . Equality in matters of public employment : Article 16 of the constitution lays down that the State can not
discriminate against anyone in the matters of employment . All citizens can apply for government jobs . There are some exceptions .
</item 7>

<item 8>
Fundamental rights in India
In 2007 , Supreme Court ruled that there could not be any blanket immunity from judicial review for the laws inserted in the Ninth Schedule . Apex court also
stated it shall examine laws included in the Ninth Schedule after 1973 for any incompatibility with the basic structure doctrine . Amendment to Article 31C (
edit ) Section 4 of the 42nd Amendment , had changed Article 31C of the constitution to accord precedence to the Directive Principles ( earlier applicable only
to clauses b & c of Article 39 ) over the fundamental rights of individuals . In Minerva Mills v. Union of India case , supreme court ruled that the amendment
to the Article 31C is not valid and ultra vires . Right to property ( edit ) The Constitution originally provided for the right to property under Articles 19
and 31 . Article 19 guaranteed to all citizens the right to acquire , hold and dispose of property . Article 31 provided that `` no person shall be deprived of
his property save by authority of law . '' It also provided that compensation would be paid to a person whose property has been taken for public purposes .
</item 8>

<item 9>
Fundamental rights in India
Jump up ^ Sovereign means possessing supreme or ultimate power . `` in modern democracies the people 's will is in theory sovereign '' synonyms : supreme ,
absolute , unlimited , unrestricted , unrestrained , unbounded , boundless , infinite , ultimate , total , unconditional , full , utter , paramount ; principal
, chief , dominant , predominant Jump up ^ `` What causes judicial delay ? Judgements diluting time frames in Code of Civil Procedure worsen the problem of
adjournments '' . Retrieved 13 September 2017 . Jump up ^ `` Paras 316 and 317 of Kesavananda Bharati v. State of Kerala , ( AIR 1973 SC 1461 ) '' . 1973 .
Retrieved 21 October 2016 . Jump up ^ `` IX Schedule laws open to review '' . 2007 . Retrieved 21 October 2017 . Jump up ^ 44th Amendment Act , 1978 . Jump up ^
Mahapatra , Dhananjay ( 28 February 2009 ) . `` Should right to property return ? '' . The Times of India . Retrieved 8 July 2010 .
</item 9>

<item 10>
Fundamental rights in India
Article 1 of Part 1 of the Indian constitution , defines India ( Bharat ) as union of states . In nutshell , India is its people not its land as enshrined in
its constitution . Since speedy trial is not the constitutional right of the citizens , the cases involving violations of fundamental rights take inordinate
time for resolution by the Supreme Court which is against the legal maxim ' justice delayed is justice denied ' . Amendments ( edit ) Changes to the fundamental
rights require a constitutional amendment , which has to be passed by a special majority of both houses of Parliament . This means that an amendment requires
the approval of two - thirds of the members present and voting . However , the number of members voting in support of the amendment shall not be less than the
simple majority of the total members of a house -- whether the Lok Sabha or Rajya Sabha . Fundamental rights not sacrosanct ( edit ) While deciding the
Golaknath case in February 1967 , Supreme Court ruled that the Parliament has no power to curtail the fundamental rights .
</item 10>
=== END ITEM id=851c658c3c421e91 ===
