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

=== ITEM id=d5647b9610c9c2ba ===
Gold sentences and facts:
Sentence 0: In the aftermath of the disputed 1824 presidential election , the separate factions took on many characteristics of parties in their own right . Adams ' supporters , in league with Clay , favored modernization , banks , industrial development and federal spending for roads and other internal improvements , which the Old Republicans and the Jackson men usually opposed . Writing in his personal journal on December 13 , 1826 , President Adams noted the difficulty he faced in attempting to be nonpartisan in appointing men to office :
  Fact 0.0: After the disputed 1824 presidential election, the separate factions took on many characteristics of parties in their own right.
  Fact 0.1: Adams' supporters, in league with Clay, favored modernization.
  Fact 0.2: Adams' supporters, in league with Clay, favored banks.
  Fact 0.3: Adams' supporters, in league with Clay, favored industrial development.
  Fact 0.4: Adams' supporters, in league with Clay, favored federal spending for roads and other internal improvements.
  Fact 0.5: The Old Republicans and the Jackson men usually opposed these positions.
  Fact 0.6: President Adams wrote in his personal journal on December 13, 1826.
  Fact 0.7: President Adams noted in his personal journal on December 13, 1826 the difficulty he faced in attempting to be nonpartisan in appointing men to office.

Retrieved items:
<item 1>
Democratic-Republican Party
On rare occasions a Senator resigned rather than follow instructions . The opposition Federalist Party quickly declined , suffering from a lack of leadership
after the death of Hamilton and the retirement of John Adams . It revived briefly in opposition to the War of 1812 , but the extremism of its Hartford
Convention of 1815 utterly destroyed it as a political force . National debt ( edit ) Jefferson and Albert Gallatin focused on the danger that the public debt ,
unless it was paid off , would be a threat to republican values . They were appalled that Hamilton was increasing the national debt and using it to solidify his
Federalist base . Gallatin was the Republican Party 's chief expert on fiscal issues and as Treasury Secretary under Jefferson and Madison worked hard to lower
taxes and lower the debt , while at the same time paying cash for the Louisiana Purchase and funding the War of 1812 .
</item 1>

<item 2>
Democratic-Republican Party
Jefferson and Madison often used the terms `` republican '' and `` Republican party '' in their letters . The 1804 Convention of Republican members of Congress
that renominated Jefferson described itself as a `` regular republican caucus '' . The name Democratic - Republican was used by contemporaries only occasionally
. The term `` republican '' was in widespread usage from the 1770s to describe the type of government the break - away colonies wanted to form : a republic of
three separate branches of government derived from some principles and structure from ancient republics ; especially the emphasis on civic duty and the
opposition to corruption , elitism , aristocracy and monarchy . The word is used in the United States Constitution . Legacy ( edit ) Further information : First
Party System and Second Party System A split appeared in the then Republican party during the 1824 elections ( at the end of the Monroe administration ) . When
the election was thrown to the House of Representatives , Henry Clay backed John Quincy Adams to deny the presidency to Andrew Jackson , a longtime political
rival . Jackson defeated Adams in 1828 and in the next election the first Democratic national convention took place in Baltimore , Maryland on May 21 -- 23 ,
1832 .
</item 2>

<item 3>
Democratic-Republican Party
Thomas Jefferson wrote on the state of party politics in the early 1820s : An opinion prevails that there is no longer any distinction , that the republicans &
Federalists are completely amalgamated but it is not so . The amalgamation is of name only , not of principle . All indeed call themselves by the name of
Republicans , because that of Federalists was extinguished in the battle of New Orleans . But the truth is that finding that monarchy is a desperate wish in
this country , they rally to the point which they think next best , a consolidated government . Their aim is now therefore to break down the rights reserved by
the constitution to the states as a bulwark against that consolidation , the fear of which produced the whole of the opposition to the constitution at its birth
. Hence new Republicans in Congress , preaching the doctrines of the old Federalists , and the new nick - names of Ultras and Radicals . But I trust they will
fail under the new , as the old name , and that the friends of the real constitution and union will prevail against consolidation , as they have done against
monarchism .
</item 3>

<item 4>
Democratic-Republican Party
Jefferson came in second in the electoral college ( at the time , its balloting could not distinguish between President and Vice President ) and became Vice
President . He would become a consistent and strong opponent of the policies of the John Adams administration . Jefferson and Madison were deeply upset by the
unconstitutionality of the Alien and Sedition Acts of 1798 and they secretly wrote the Kentucky and Virginia Resolutions , which called on state legislatures to
nullify unconstitutional laws . However , the other states did not follow suit and several rejected the notion that states could nullify federal law . The
Republican critique of federalism became wrapped in the slogan of `` Principles of 1798 '' , which became the hallmark of the party . The most important of
these principles were states ' rights , opposition to a strong national government , distrust of the federal courts and opposition to the navy and the national
bank . The party saw itself as a champion of republicanism and denounced the Federalists as supporters of monarchy and aristocracy . The party coalesced around
Jefferson , who diligently maintained extensive correspondence with like - minded Republican leaders throughout the country .
</item 4>

<item 5>
Democratic-Republican Party
I scarcely know myself which is most to be deprecated , a consolidation , or dissolution of the states . The horrors of both are beyond the reach of human
foresight . In the aftermath of the disputed 1824 presidential election , the separate factions took on many characteristics of parties in their own right .
Adams ' supporters , in league with Clay , favored modernization , banks , industrial development and federal spending for roads and other internal improvements
, which the Old Republicans and the Jackson men usually opposed . Writing in his personal journal on December 13 , 1826 , President Adams noted the difficulty
he faced in attempting to be nonpartisan in appointing men to office : And it is upon the occasion of appointments to office that all the wormwood and the gall
of the old party hatred ooze out . Not a vacancy to any office occurs but there is a distinguished federalist started and pushed home as a candidate to fill it
-- always well qualified , sometimes in an eminent degree , and yet so obnoxious to the Republican party that he can not be appointed without exciting a
vehement clamor against him and against the Administration .
</item 5>

<item 6>
Democratic-Republican Party
Politicians ever since have used the issue of a high national debt to denounce the other party for profligacy and a threat to fiscal soundness and the nation 's
future . Monroe and Adams ( 1816 -- 1828 ) ( edit ) In rapidly expanding western states , the Federalists had few supporters . Every state had a distinct
political geography that shaped party membership . In Pennsylvania , the Republicans were weakest around Philadelphia and strongest in Scots - Irish settlements
in the west . Members came from all social classes , but came predominantly from the poor , subsistence farmers , mechanics and tradesmen . After the War of
1812 , partisanship subsided across the young republic -- people called it the Era of Good Feelings . James Monroe narrowly won the party 's nomination for
President in Congress over William Crawford in 1816 and defeated Federalist Rufus King in the general election . In the early years of the party , the key
central organization grew out of caucuses of Congressional leaders in Washington . However , the key battles to choose electors occurred in the states , not in
the caucus .
</item 6>

<item 7>
Democratic-Republican Party
Washington frequently decried the growing sense of `` party '' emerging from the internal battles among Jefferson , Madison , Hamilton , Adams and others in his
administration . As warfare in Europe increased , the two factions increasingly made foreign policy the central political issue of the day . The Republicans
wanted to maintain the 1778 alliance with France , which had overthrown the monarchy and aristocracy and become a republic . Even though Britain was by far
United States ' leading trading partner , Republicans feared that increased trade would undermine republicanism . The Republicans distrusted Hamilton 's
national bank and rejected his premise that a national debt was good for the country as Republicans said they were both forms of corruption . They strongly
distrusted the elitism of Hamilton 's circle , denouncing it as `` aristocratic '' ; and they called for states ' rights lest the Federalists centralize ever
more power in the national governments . The intense debate over the Jay Treaty in 1794 -- 1795 transformed those opposed to Hamilton 's policies from a loose
movement into a true political party .
</item 7>

<item 8>
Democratic-Republican Party
these were slowly coalescing groups with initially considerable independent thinking and voting . Cunningham noted that only about a quarter of the House of
Representatives up until 1794 voted with Madison as much as two - thirds of the time and another quarter against him two - thirds of the time , leaving almost
half as fairly independent . Albert Gallatin recalled only two caucuses on legislative policy between 1795 and 1801 , one over appropriations for Jay 's Treaty
and the other over the Quasi-War , but in neither case did the party decide to vote unanimously . Organizational strategy ( edit ) The new party invented some
of the campaign and organizational techniques which were later adopted by the Federalists and became standard American practice . It was especially effective in
building a network of newspapers in major cities to broadcast its statements and editorialize its policies . Fisher Ames , a leading Federalist , used the term
`` Jacobin '' to link members of Jefferson 's party to the radicals of the French Revolution .
</item 8>

<item 9>
Democratic-Republican Party
Foreign affairs took a leading role in 1794 -- 1795 as the Republicans vigorously opposed the Jay Treaty with Britain , which was then at war with France .
Republicans saw France as more democratic after its revolution , while Britain represented the hated monarchy . The party denounced many of Hamilton 's measures
as unconstitutional , especially the national bank . The party was strongest in the South and weakest in the Northeast . It demanded states ' rights as
expressed by the `` Principles of 1798 '' articulated in the Kentucky and Virginia Resolutions that would allow states to nullify a federal law . Above all ,
the party stood for the primacy of the yeoman farmers . Republicans were deeply committed to the principles of republicanism , which they feared were threatened
by the supposed monarchical tendencies of the Hamiltonian Federalists . The party came to power in 1801 with the election of Jefferson in the 1800 presidential
election . The Federalists -- too elitist to appeal to most people -- faded away and totally collapsed after 1815 . Despite internal divisions , the Republicans
dominated the First Party System until partisanship itself withered away during the Era of Good Feelings after 1816 .
</item 9>

<item 10>
Democratic-Republican Party
It began in 1791 as one faction in Congress and included many politicians who had been opposed to the new constitution . They called themselves `` Republicans
'' after their ideology , republicanism . They distrusted the Federalist commitment to republicanism . The party splintered in 1824 into the Jacksonian movement
( which became the Democratic Party in 1828 ) and the short - lived National Republican Party ( later succeeded by the Whig Party ) . The term `` Democratic -
Republican '' is used especially by modern political scientists for the first `` Republican Party '' ( as opposed to the modern Republican Party founded in 1854
) . It is also known as the Jeffersonian Republicans . Historians typically use the title `` Republican Party '' . An `` Anti-Administration '' faction met
secretly in the national capital ( Philadelphia ) to oppose Hamilton 's financial programs . Jefferson denounced the programs as leading to monarchy and
subversive of republicanism . Jefferson needed to have a nationwide party to challenge the Federalists , which Hamilton was building up with allies in major
cities .
</item 10>
=== END ITEM id=d5647b9610c9c2ba ===

=== ITEM id=8a8e3d59d94cd22e ===
Gold sentences and facts:
Sentence 0: Spektor wrote and recorded the main title theme song , `` You 've Got Time '' , for the Netflix Original series Orange Is the New Black , which premiered in July 2013 . It was nominated in the Best Song Written for Visual Media category at the 56th Annual Grammy Awards .
  Fact 0.0: Spektor wrote the main title theme song 'You've Got Time' for the Netflix Original series Orange Is the New Black.
  Fact 0.1: Spektor recorded the main title theme song 'You've Got Time' for the Netflix Original series Orange Is the New Black.
  Fact 0.2: Orange Is the New Black premiered in July 2013.
  Fact 0.3: 'You've Got Time' was nominated in the Best Song Written for Visual Media category at the 56th Annual Grammy Awards.
Sentence 1: Regina wrote and recorded `` You 've Got Time '' to be the theme song for the Netflix Original Series Orange is the New Black which premiered in 2013 . Spektor was approached by the show 's creator , Jenji Kohan , to create the opening number .
  Fact 1.0: Regina wrote 'You've Got Time' to be the theme song for the Netflix Original Series Orange is the New Black.
  Fact 1.1: Regina recorded 'You've Got Time' to be the theme song for the Netflix Original Series Orange is the New Black.
  Fact 1.2: Orange is the New Black premiered in 2013.
  Fact 1.3: Spektor was approached by the show's creator, Jenji Kohan, to create the opening number.

Retrieved items:
<item 1>
Crazy Eyes (character)
( October 2 , 2013 ) . `` Spot the difference ! Orange Is The New Black star Taylor Schilling brings the real Piper Chapman as her red carpet date '' . Daily
Mail . Retrieved October 4 , 2014 . Jump up ^ Fitzgerald , Sean ( June 23 , 2014 ) . `` ' Orange is the New Black ' : Meet Crazy Eyes '' . Toronto Sun .
Retrieved October 23 , 2014 . Jump up ^ Leopold , Todd ( June 6 , 2014 ) . `` ' Orange is the New Black ' : Five things to expect from season 2 '' . CNN .
Retrieved September 6 , 2014 . Jump up ^ Hooton , Christopher ( June 4 , 2014 ) . `` Orange Is The New Black : Season 1 recap ahead of the new episodes '' . The
Independent . Retrieved September 6 , 2014 . Jump up ^ Schaefer , Megan ( June 3 , 2014 ) .
</item 1>

<item 2>
Crazy Eyes (character)
Deadline.com . Retrieved September 5 , 2014 . Jump up ^ Kohan , Jenji ( writer / creator ) ; Herrmann , Tara ( producer ) ; Burley , Mark ( producer ) ( May 13
, 2014 ) . Orange Is the New Black : Season 1 -- `` I Was n't Ready '' audio commentary ( Blu - ray ) . Lionsgate . Event occurs at 37 : 00 -- 37 : 30 . Jump up
^ `` Uzo Aduba '' . Internet Movie Database . Retrieved June 12 , 2015 . Jump up ^ `` Orange is the New Black Season 2 Episode 3 : Hugs Can Be Deceiving '' .
TV.com . Retrieved September 5 , 2014 . Jump up ^ Potts , Kimberly ( June 18 , 2014 ) . `` The 10 Best ' Orange Is the New Black ' Character Backstories '' .
Yahoo ! TV . Retrieved September 5 , 2014 . Jump up ^ White , Chelsea F.C. ( October 2 , 2013 ) . `` Spot the difference !
</item 2>

<item 3>
Crazy Eyes (character)
TV.com . Retrieved September 6 , 2014 . Jump up ^ Reed , Alex ( July 22 , 2014 ) . `` ' Orange Is the New Black ' Star Uzo Aduba on ' Crazy Eyes ' : ' People
Underestimate Her ' ( Video ) '' . The Wrap . Retrieved September 6 , 2014 . Jump up ^ Surette , Tim . `` Orange Is the New Black Season 2 Review , Part 2 (
Episodes 7 - 13 ) : Fight the Power '' . TV.com . Retrieved September 6 , 2014 . Jump up ^ Surette , Tim . `` Orange Is the New Black Season 2 Review , Part 1 (
Episodes 1 -- 6 ) : Conflict Over Community '' . TV.com . Retrieved September 6 , 2014 . Jump up ^
http://www.bustle.com/articles/90452-who-is-crazy-eyes-love-interest-on-orange-is-the-new-black-emily-althaus-character-has Jump up ^ Harnick , Chris ( June 18
, 2016 ) .
</item 3>

<item 4>
Suzanne “Crazy Eyes” Warren, played by Uzo Aduba in *Orange Is the New Black*, is intelligent and imaginative but socially isolated and emotionally volatile.
Longing for friendship, she becomes fixated on Piper, then is manipulated by Vee; later she writes a popular science-fiction erotica series and dates Maureen
Kukudio. Her imprisonment stems from accidentally causing a child’s death after inviting him home, unaware he was frightened. Aduba won Emmy awards for both
comedy and
</item 4>

<item 5>
Regina Spektor
Her world tour in support of What We Saw from the Cheap Seats included a performance in Moscow ; Spektor had not yet returned since fleeing with her family in
1989 . In 2012 , Spektor was christened an official `` Steinway Artist '' ; she plays Steinway & Sons pianos almost exclusively . Spektor wrote and recorded the
main title theme song , `` You 've Got Time '' , for the Netflix Original series Orange Is the New Black , which premiered in July 2013 . It was nominated in
the Best Song Written for Visual Media category at the 56th Annual Grammy Awards . 2016 -- present : Remember Us to Life ( edit ) Spektor announced her seventh
album , Remember Us to Life on July 21 , 2016 through her email newsletter . The album was released on September 30 , 2016 . The first single , `` Bleeding
Heart '' , was released July 22 , 2016 . The follow - up single , `` Small Bill $ '' , was released August 11 , 2016 .
</item 5>

<item 6>
Regina Spektor
The title of the song `` Do n't Leave Me ( Ne Me Quitte Pas ) '' and the cover of What We Saw from the Cheap Seats was featured on the display of the 5th
generation iPod Touch in promotional content from Apple in late 2012 . `` Do n't Leave Me ( Ne Me Quitte Pas ) '' was used at the end of the episode `` Five
Miles From Yetzer Hara '' from the eighth season of Weeds . The Consequence of Sound website takes its name from the Spektor song , `` Consequence of Sounds ''
. `` All the Rowboats '' was featured on The CW 's Ringer in March 2012 . The song `` Your Honor '' was used in the Season 2 premiere of HBO 's series Girls on
January 13 , 2013 , and the song `` Laughing With '' was featured on BBC drama The Crash in March 2013 as well as in HBO 's The Leftovers in November 2015 .
Regina wrote and recorded `` You 've Got Time '' to be the theme song for the Netflix Original Series Orange is the New Black which premiered in 2013 .
</item 6>

<item 7>
Crazy Eyes (character)
Jump up ^ Schaefer , Megan ( June 3 , 2014 ) . `` ' Orange Is The New Black ' Season 1 Recap : Everything To Know Before The Season 2 Premiere '' .
International Business Times . Retrieved September 6 , 2014 . Jump up ^ Foster , Lewis ( June 19 , 2014 ) . `` Visual Summary : Orange Is The New Black Season 1
'' . Paste . Retrieved September 6 , 2014 . Jump up ^ O'Sullivan , Erin ( August 14 , 2013 ) . `` Orange Is The New Black : Uzo Aduba ( Suzanne Warren )
Promoted To Series Regular '' . Access Hollywood . Retrieved September 6 , 2014 . Jump up ^ Henderson , Danielle ( August 16 , 2013 ) . `` Orange Is the New
Black Episode Nine Recap : We Finally Experience the SHU '' . Vulture . Retrieved September 6 , 2014 . Jump up ^ `` Orange is the New Black Season 1 Episode 9 :
F * cksgiving : Episode Cast & Crew '' . TV.com .
</item 7>

<item 8>
NBC Sunday Night Football
Singer Pink sang the theme song for the broadcast in 2006 , a reworking of the Joan Jett song `` I Hate Myself for Loving You '' retitled `` Waiting All Day for
Sunday Night '' . Several alternative versions were used throughout the season , substituting different lyrics when appropriate ( such as `` Waiting All Year
For Opening Night '' ) . In 2007 , country singer Faith Hill replaced Pink as the singer of the opening theme , and a new arrangement of the Joan Jett song
coincided with her debut . The Faith Hill intro , in particular , was not without criticism and parody . The intro was lampooned in the October 9 , 2010 episode
of Saturday Night Live , with host Jane Lynch as Hill ( with Jason Sudeikis as Al Michaels and Bill Hader as Cris Collinsworth ) . In the 30 Rock episode ``
Season 4 '' , the character of Jenna Maroney ( Jane Krakowski ) sings what appears to be an allusion to the Faith Hill intro for NBC 's fictional Tennis Night
in America program .
</item 8>

<item 9>
NBC Sunday Night Football
A new Underwood - sung theme known as `` Oh , Sunday Night '' , which takes elements from her 2014 duet hit with Miranda Lambert , `` Somethin ' Bad '' ,
premiered with the opening game of the 2016 season on September 11 . For NBC 's coverage of Thursday Night Football , a capella group Pentatonix sings the
opening theme song called `` Weekend Go . '' The song is a reworked version of their original song `` Sing . '' NBC also used the `` TNF '' main theme music
called `` Ca n't Hold Us Down , '' performed by members of the Broadway orchestra for Hamilton . Show opening ( edit ) The song is at the centerpiece of the
opening montage , which has changed several times over the years . Underwood 's music has always played in the background over the official welcome after the
opening is completed and the teams take the field . 2006 ( edit ) For the first season , Pink appeared to sing from the top of a skyscraper as a helicopter
zoomed down on a city skyline with enlarged players Shaun Alexander , LaDainian Tomlinson and Tom Brady and the field , the results of computer - generated
imagery .
</item 9>

<item 10>
Crazy Eyes (character)
`` Orange Is the New Black Reveals Crazy Eyes ' Tragic Crime '' . ENews . Retrieved 18 September 2017 . Jump up ^ Meltzer , Tom ( December 17 , 2013 ) . `` The
best TV of 2013 : No 6 -- Orange is the New Black ( Netflix ) : The US prison drama -- Netflix 's most - watched show of the year -- is a binge - watcher 's
dream : moreish and full of surprises '' . The Guardian . Retrieved October 5 , 2014 . Jump up ^ Jancelewicz , Chris ( July 3 , 2014 ) . `` Orange Is The New
Black Review : Time Behind Bars Should n't Be This Fun '' . The Huffington Post Canada . Retrieved October 6 , 2014 . Jump up ^ `` Primetime Emmy Awards 2014 :
The winners list '' . CNN . August 26 , 2014 . Retrieved September 4 , 2014 . Jump up ^ `` The International Press Academy Announces Nominations For The 18th
Annual Satellite Awards '' . PR Newswire .
</item 10>
=== END ITEM id=8a8e3d59d94cd22e ===

=== ITEM id=d30119a84fdec915 ===
Gold sentences and facts:
Sentence 0: `` Sacrifice '' is a ballad performed by musician Elton John . The lyrics are by Bernie Taupin and the music by John . The song appears on the 1989 album Sleeping with the Past . It was first released in October 1989 , then in 1990 , and was the second single from the album . It achieved success , particularly in France and the UK , where it became his first solo chart - topper , spending five weeks at the top .
  Fact 0.0: 'Sacrifice' is a ballad performed by musician Elton John.
  Fact 0.1: The lyrics of 'Sacrifice' are by Bernie Taupin.
  Fact 0.2: The music of 'Sacrifice' is by Elton John.
  Fact 0.3: 'Sacrifice' appears on the 1989 album Sleeping with the Past.
  Fact 0.4: 'Sacrifice' was first released in October 1989.
  Fact 0.5: 'Sacrifice' was released in 1990.
  Fact 0.6: 'Sacrifice' was the second single from the album Sleeping with the Past.
  Fact 0.7: 'Sacrifice' achieved success, particularly in France and the UK.
  Fact 0.8: 'Sacrifice' became Elton John's first solo chart-topper in France and the UK.
  Fact 0.9: 'Sacrifice' spent five weeks at the top of the chart.

Retrieved items:
<item 1>
Sacrifice (song)
The song is beloved by both John and Bernie Taupin and deemed a bookend of sorts to their first hit `` Your Song '' . Taupin has said he is surprised he wrote
it , and said in an interview , `` I think ' Sacrifice ' is one of the best songs we 've written . '' The song is not a typical love song , but rather a song
about a breakup of a marriage where the loss of the relationship is `` no sacrifice . '' Although the single was successful , Elton John did not tour Europe in
both 1989 -- 1990 to support the Sleeping with the Past album ( with the exception of one appearance in Knebworth Park on 30 June 1990 ) touring only in the
U.S. and Australia . The 1989 European tour was , instead , in support of the Reg Strikes Back album , with no material representing Sleeping with the Past . In
all shows that were done to support Sleeping with the Past in 1989 -- 1990 , Elton played this song solo on his Roland RD - 1000 Digital Piano .
</item 1>

<item 2>
Sacrifice (song)
The lyrics are by Bernie Taupin and the music by John . The song appears on the 1989 album Sleeping with the Past . It was first released in October 1989 , then
in 1990 , and was the second single from the album . It achieved success , particularly in France and the UK , where it became his first solo chart - topper ,
spending five weeks at the top . A country music cover of the song was performed by Don Henley and Vince Gill as part of the 2018 tribute album Restoration :
Reimagining the Songs of Elton John and Bernie Taupin . Contents ( hide ) 1 Overview 1.1 Background and writing 1.2 Music video 1.3 Chart performance 2 Track
listings 3 Charts and sales 3.1 Weekly charts 3.2 Year - end charts 3.3 Certifications and sales 4 Personnel 5 References 6 External links Overview ( edit )
Background and writing ( edit ) `` Sacrifice '' is inspired both by the Aretha Franklin song `` Do Right Woman , Do Right Man '' and described as a Percy Sledge
song by Elton John .
</item 2>

<item 3>
Sacrifice (song)
Awarded as The Best Song Retrieved from `` https://en.wikipedia.org/w/index.php?title=Sacrifice_(song)&oldid=833955469 '' Categories : 1989 singles 1990 singles
Disney songs Elton John songs European Hot 100 Singles number - one singles SNEP Top Singles number - one singles UK Singles Chart number - one singles Songs
with lyrics by Bernie Taupin Songs with music by Elton John Pop ballads Rock ballads Song recordings produced by Chris Thomas ( record producer ) 1989 songs MCA
Records singles The Rocket Record Company singles Music videos directed by Alek Keshishian Hidden categories : CS1 Dutch - language sources ( nl ) CS1 French -
language sources ( fr ) Webarchive template wayback links CS1 Norwegian - language sources ( no ) CS1 German - language sources ( de ) Articles with hAudio
microformats Use dmy dates from August 2010 Talk Contents About Wikipedia Dansk Español Français Italiano Nederlands Norsk nynorsk Polski Português Edit links
This page was last edited on 3 April 2018 , at 05 : 22 .
</item 3>

<item 4>
Graph query row:
song: Sacrifice
relationship: PART_OF
album: Sleeping with the Past
</item 4>

<item 5>
Sacrifice (song)
Sacrifice ( song ) - wikipedia Sacrifice ( song ) Jump to : navigation , search For other songs with the same title , see Sacrifice ( disambiguation ) § Songs .
`` Sacrifice '' Single by Elton John from the album Sleeping with the Past A-side `` Healing Hands '' ( second release ) B - side `` Love Is a Cannibal '' (
first release ) Released October 1989 ( first release ) May 1990 ( second release ) Format CD , vinyl record ( 7 '' and 12 `` ) , audio cassette Length 5 : 11
Label Rocket , MCA Songwriter ( s ) Elton John , Bernie Taupin Producer ( s ) Chris Thomas Elton John singles chronology `` Healing Hands '' ( 1989 ) ``
Sacrifice '' ( 1989 ) `` Club at the End of the Street '' ( 1990 ) `` Healing Hands '' ( 1989 ) `` Sacrifice '' ( 1989 ) `` Club at the End of the Street '' (
1990 ) `` Sacrifice '' is a ballad performed by musician Elton John . The lyrics are by Bernie Taupin and the music by John .
</item 5>

<item 6>
Sacrifice (song)
'' ( 1997 ) `` Something About the Way You Look Tonight '' ( 1997 ) `` Candle in the Wind 1997 '' ( 1997 ) `` Written in the Stars '' ( 1999 ) 2000s `` I Want
Love '' ( 2001 ) `` This Train Do n't Stop There Anymore '' ( 2002 ) `` Original Sin '' ( 2002 ) `` Are You Ready for Love '' ( 2003 ) `` Ghetto Gospel '' (
2005 ) `` Electricity '' ( 2005 ) `` The Bridge '' ( 2006 ) `` Joseph , Better You than Me '' ( 2008 ) `` Tiny Dancer ( Hold Me Closer ) '' ( 2009 ) Related
articles Albums discography Videography Book : Elton John Category : Elton John Ivor Novello Awards -- Best Song Musically and Lyrically 1960s She 's Leaving
Home ( 1968 ) 1970s Where Do You Go To ( My Lovely ) ? ( 1970 ) Something ( 1971 ) Do n't Let It Die ( 1972 ) Without You ( 1973 ) Daniel ( 1974 ) Streets
</item 6>

<item 7>
Sacrifice (song)
Jump up ^ `` UK certifications , database '' . Bpi . Retrieved 9 September 2008 . External links ( edit ) Lyrics of this song at MetroLyrics Elton John songs
1960s and 1970s `` Skyline Pigeon '' ( 1968 ) `` I 've Been Loving You '' ( 1968 ) `` Lady Samantha '' ( 1969 ) `` It 's Me That You Need '' ( 1969 ) `` Border
Song '' ( 1970 ) `` Rock and Roll Madonna '' ( 1970 ) `` From Denver to L.A. '' ( 1970 ) `` Take Me to the Pilot '' ( 1970 ) `` Your Song '' ( 1970 ) `` Friends
'' ( 1971 ) `` Levon '' ( 1971 ) `` Tiny Dancer '' ( 1972 ) `` Rocket Man '' ( 1972 ) `` Honky Cat '' ( 1972 ) `` I Think I 'm Going to Kill Myself '' ( 1972 )
`` Mona Lisas and Mad Hatters '' ( 1972 ) `` Crocodile Rock '' ( 1972 ) `` Elderberry Wine '' (
</item 7>

<item 8>
Sacrifice (song)
Billboard Hot Adult Contemporary Tracks Chart ( 1990 ) Peak position Australian ARIA Singles Chart 7 Dutch Mega Top 100 Eurochart Hot 100 French SNEP Singles
Chart German Singles Chart 36 Irish Singles Chart New Zealand RIANZ Singles Chart 7 Norwegian Singles Chart Swiss Singles Chart 23 UK Singles Chart U.S.
Billboard Hot 100 18 Zimbabwe Singles Chart 6 Year - end charts ( edit ) Chart ( 1990 ) Rank Australian Singles Chart 47 Dutch Top 40 11 UK Singles Charts
Certifications and sales ( edit ) Country Certification Date Sales certified Physical sales France Silver 200,000 380,000 UK Platinum 1 September 1990 600,000
Personnel ( edit ) Elton John : vocals ,
</item 8>

<item 9>
Sacrifice (song)
000 Personnel ( edit ) Elton John : vocals , Roland RD - 1000 digital piano Davey Johnstone : electric guitar Romeo Williams : bass Jonathan Moffett : drums Guy
Babylon : keyboards Fred Mandel : keyboards Preceded by `` Words '' by The Christians French SNEP number - one single 26 May 1990 -- 9 June 1990 ( 3 weeks )
Succeeded by `` Bo le lavabo ( WC Kiss ) '' by Lagaf ' Preceded by `` Vogue '' by Madonna Eurochart Hot 100 number - one single 16 June 1990 -- 28 July 1990 ( 7
weeks ) Succeeded by `` U Ca n't Touch This '' by MC Hammer Preceded by `` World In Motion '' by Englandneworder UK number - one single 17 June 1990 -- 21 July
1990 ( 5 weeks ) Succeeded by `` Turtle Power '' by Partners in Kryme References ( edit ) ^ Jump up to : `` Elton John - Full Official Chart History '' .
Official Charts Company . Official Charts Company . Retrieved 20 January 2016 .
</item 9>

<item 10>
Sacrifice (song)
Thus , it became John 's first solo number - one single in the UK Singles Chart , remaining on the top spot for five weeks . With this re-release , Elton John
also got his first number one in France and stayed on the chart for 26 weeks . Track listings ( edit ) First release 12 '' single -- UK `` Sacrifice '' -- 5 :
07 `` Love Is a Cannibal '' ( From the Columbia Motion Picture Ghostbusters II ) -- 3 : 53 CD single -- U.S. `` Sacrifice '' -- 5 : 07 `` Love Is a Cannibal ''
( from the Columbia Motion Picture Ghostbusters II ) -- 3 : 53 `` Durban Deep '' Second release UK single `` Sacrifice '' -- 5 : 07 `` Healing Hands '' -- 4 :
22 Charts and sales ( edit ) Weekly charts ( edit ) Chart ( 1989 ) Peak position UK Singles Chart 55 U.S.
</item 10>
=== END ITEM id=d30119a84fdec915 ===

=== ITEM id=cc8940c900bead20 ===
Gold sentences and facts:
Sentence 0: The Proud Family is a soundtrack album for the show of the same name . A combination of both original tunes by the characters in the show , popular hits by popular music artists such as Alicia Keys , Jhené Aiko , India Arie , and Solange Knowles & Destiny 's Child ( who perform the theme song ) , and classic soul music from artists such as Aretha Franklin and The O'Jays .
  Fact 0.0: The Proud Family is a soundtrack album.
  Fact 0.1: The Proud Family soundtrack album is for the show of the same name.
  Fact 0.2: The album combines original tunes by the characters in the show.
  Fact 0.3: The album combines popular hits by popular music artists such as Alicia Keys, Jhené Aiko, India Arie, and Solange Knowles & Destiny's Child.
  Fact 0.4: Solange Knowles & Destiny's Child perform the theme song.
  Fact 0.5: The album combines classic soul music from artists such as Aretha Franklin and The O'Jays.

Retrieved items:
<item 1>
The Partridge Family
The Partridge Family - wikipedia The Partridge Family Jump to : navigation , search The Partridge Family Genre Musical sitcom Created by Bernard Slade Starring
Shirley Jones David Cassidy Danny Bonaduce Susan Dey Suzanne Crough Jeremy Gelbwaks Dave Madden Brian Forster Theme music composer Diane Hilderbrand Danny
Janssen Wes Farrell Opening theme `` When We 're Singin ' '' ( 1970 -- 71 ) `` C'mon , Get Happy '' ( 1971 -- 74 ) Composer ( s ) George Duning Benny Golson
Warren Barker Hugo Montenegro Shorty Rogers Country of origin United States Original language ( s ) English No. of seasons No. of episodes 96 ( list of episodes
) Production Executive producer ( s ) Bob Claver Producer ( s ) William Bickley Paul Junger Witt Dale McRaven Larry Rosen Mel Swope Cinematography Fred Jackman
, Jr .
</item 1>

<item 2>
NBC Sunday Night Football
A new Underwood - sung theme known as `` Oh , Sunday Night '' , which takes elements from her 2014 duet hit with Miranda Lambert , `` Somethin ' Bad '' ,
premiered with the opening game of the 2016 season on September 11 . For NBC 's coverage of Thursday Night Football , a capella group Pentatonix sings the
opening theme song called `` Weekend Go . '' The song is a reworked version of their original song `` Sing . '' NBC also used the `` TNF '' main theme music
called `` Ca n't Hold Us Down , '' performed by members of the Broadway orchestra for Hamilton . Show opening ( edit ) The song is at the centerpiece of the
opening montage , which has changed several times over the years . Underwood 's music has always played in the background over the official welcome after the
opening is completed and the teams take the field . 2006 ( edit ) For the first season , Pink appeared to sing from the top of a skyscraper as a helicopter
zoomed down on a city skyline with enlarged players Shaun Alexander , LaDainian Tomlinson and Tom Brady and the field , the results of computer - generated
imagery .
</item 2>

<item 3>
Yo Gabba Gabba!
( 2009 ) Yo Gabba Gabba ! ( 2010 ) Sesame Street ( 2011 ) Switched at Birth ( 2012 ) Bunheads ( 2013 ) The Fosters ( 2014 ) The Fosters ( 2015 ) Daniel Tiger 's
Neighborhood ( 2016 ) Speechless ( 2017 ) MusicBrainz : fd649d06 - c1bc - 4c95 - 8c48 - eb7b91726e65 Retrieved from ``
https://en.wikipedia.org/w/index.php?title=Yo_Gabba_Gabba!&oldid=837777521 '' Categories : 2007 American television series debuts 2000s American music
television series 2015 American television series endings 2010s American music television series American children 's musical television series English -
language television programs American television programs featuring anthropomorphic characters American television series with live action and animation
Television series by DHX Media Animated musical television series Nick Jr.
</item 3>

<item 4>
The Proud Family (soundtrack)
The Proud Family ( soundtrack ) - Wikipedia The Proud Family ( soundtrack ) Jump to : navigation , search This article does not cite any sources . Please help
improve this article by adding citations to reliable sources . Unsourced material may be challenged and removed . ( December 2009 ) ( Learn how and when to
remove this template message ) The Proud Family Soundtrack album by Various Artists Released April 20 , 2004 August 16 , 2005 ( Re-release ) Genre Soul , hip
hop , hip hop soul Label Walt Disney The Proud Family is a soundtrack album for the show of the same name . A combination of both original tunes by the
characters in the show , popular hits by popular music artists such as Alicia Keys , Jhené Aiko , India Arie , and Solange Knowles & Destiny 's Child ( who
perform the theme song ) , and classic soul music from artists such as Aretha Franklin and The O'Jays . References ( edit ) This soundtrack - related article is
a stub . You can help Wikipedia by expanding it .
</item 4>

<item 5>
The Partridge Family
More than one song was used as the theme song . Season 1 episodes feature the song `` When We 're Singin ' '' , which was replaced in subsequent seasons with ``
C'mon Get Happy '' . `` Five of us , and Mom working all day , we knew we could help her if our music would pay , Danny got Reuben to sell our song , and it
really came together when Mom sang along ... '' ( from `` When We 're Singin ' '' ) Later , under Farrell 's direction , the new version had new lyrics sung to
the `` When We 're Singin ' '' tune . With the new chorus finalized , `` C'mon Get Happy '' showcased the new verse : `` We had a dream , we 'd go travelin '
together , We 'd spread a little lovin ' then we 'd keep movin ' on .
</item 5>

<item 6>
The Proud Family (soundtrack)
You can help Wikipedia by expanding it . Retrieved from `` https://en.wikipedia.org/w/index.php?title=The_Proud_Family_(soundtrack)&oldid=638462881 ''
Categories : Television animation soundtracks 2004 soundtracks Hip hop soundtracks Soul soundtracks Walt Disney Records soundtracks Soundtrack stubs Hidden
categories : Articles lacking sources from December 2009 All articles lacking sources Articles with hAudio microformats Music infoboxes with deprecated
parameters All stub articles Talk About Wikipedia Add links This page was last edited on 17 December 2014 , at 07 : 09 . Text is available under the Creative
Commons Attribution - ShareAlike License ; additional terms may apply . By using this site , you agree to the Terms of Use and Privacy Policy . Wikipedia ® is a
registered trademark of the Wikimedia Foundation , Inc. , a non-profit organization . About Wikipedia
</item 6>

<item 7>
The Partridge Family
L-R : Shirley Jones , Jeremy Gelbwaks , Suzanne Crough , Susan Dey , Danny Bonaduce and David Cassidy In the pilot episode , a group of musical siblings in the
fictitious city of San Pueblo , California ( said to be `` 40 miles from Napa County '' in episode 24 , A Partridge By Any Other Name ) convinces their widowed
mother , bank teller Shirley Partridge , to help them out by singing as they record a pop song in their garage . Through the efforts of precocious 10 - year -
old Danny they find a manager , Reuben Kincaid , who helps make the song a Top 40 hit . After more persuading , Shirley agrees that the family can go on tour .
They acquire an old school bus , a 1957 Chevrolet Series 6800 Superior , for touring , paint it with Mondrian - inspired patterns , and head to Las Vegas ,
Nevada for their first live gig at Caesars Palace . Subsequent episodes usually feature the band performing in various venues or in their garage . The shows
often contrast suburban life with the adventures of a show - business family on the road .
</item 7>

<item 8>
NBC Sunday Night Football
Singer Pink sang the theme song for the broadcast in 2006 , a reworking of the Joan Jett song `` I Hate Myself for Loving You '' retitled `` Waiting All Day for
Sunday Night '' . Several alternative versions were used throughout the season , substituting different lyrics when appropriate ( such as `` Waiting All Year
For Opening Night '' ) . In 2007 , country singer Faith Hill replaced Pink as the singer of the opening theme , and a new arrangement of the Joan Jett song
coincided with her debut . The Faith Hill intro , in particular , was not without criticism and parody . The intro was lampooned in the October 9 , 2010 episode
of Saturday Night Live , with host Jane Lynch as Hill ( with Jason Sudeikis as Al Michaels and Bill Hader as Cris Collinsworth ) . In the 30 Rock episode ``
Season 4 '' , the character of Jenna Maroney ( Jane Krakowski ) sings what appears to be an allusion to the Faith Hill intro for NBC 's fictional Tennis Night
in America program .
</item 8>

<item 9>
You'll Never Find Another Love Like Mine
in which Lou Rawls himself sings it to Damon Wayans during a colonoscopy , That ' 70s Show ( 1998 -- 2006 ) , South Park ( 1997 -- ) , Two and a Half Men ( 2003
-- 2015 ) , and Psych ( 2006 -- 2014 ) . The song could also be heard in an episode of The Proud Family , where Lou Rawls sings and dances with Penny Proud . In
addition , it was performed by the singer in doll form in an Action League Now ! segment on KaBlam ! . The song `` If You Could Love Me '' by Edwyn Collins uses
many of the same chords , and has a similar sound . Rocco DiSpirito and Karina Smirnoff also danced the rumba to this song in week 2 of season 7 of Dancing with
the Stars .
</item 9>

<item 10>
The Partridge Family
Actor David Cassidy sang lead on most of the records , and actor Shirley Jones can be heard on a few . But the other voices and all the instruments were
recorded by session musicians . The Partridge Family 's instrumental sound was provided by musicians informally known as The Wrecking Crew , who did many of the
pop hits of the day attributed to other performers . The actors pretended to play their instruments . None of the actors sang background ; they lip - synched to
recorded singing by The Love Generation , AKA The Ron Hicklin Singers , composed of brothers John and Tom Bahler , Jackie Ward ( who as Robin Ward had a # 14
hit `` Wonderful Summer '' on Dot Records in 1963 ) , and Ron Hicklin . Actor David Cassidy was originally to lip sync with the rest of the cast , but convinced
Farrell that he could sing and was allowed to join the studio ensemble as the lead singer . He and his step - mother , Shirley , are the only two show actors
allowed to sing for recording . The Partridge Family theme song , produced by Rogers , was shown over opening credits . More than one song was used as the theme
song .
</item 10>
=== END ITEM id=cc8940c900bead20 ===

=== ITEM id=196cc6e737ba17bb ===
Gold sentences and facts:
Sentence 0: The Partridge Family Genre Musical sitcom Created by Bernard Slade Starring Shirley Jones David Cassidy Danny Bonaduce Susan Dey Suzanne Crough Jeremy Gelbwaks Dave Madden Brian Forster Theme music composer Diane Hilderbrand Danny Janssen Wes Farrell Opening theme `` When We 're Singin ' '' ( 1970 -- 71 ) `` C'mon , Get Happy '' ( 1971 -- 74 ) Composer ( s ) George Duning Benny Golson Warren Barker Hugo Montenegro Shorty Rogers Country of origin United States Original language ( s ) English No. of seasons No. of episodes 96 ( list of episodes ) Production Executive producer ( s ) Bob Claver Producer ( s ) William Bickley Paul Junger Witt Dale McRaven Larry Rosen Mel Swope Cinematography Fred Jackman , Jr . Irving Lippman Camera setup Single - camera Running time 25 minutes Production company ( s ) Screen Gems Television ( 1970 -- 1974 ) Distributor Columbia Pictures Television ( 1974 -- 84 ; 1989 -- 96 ) DFS Program Exchange ( 1984 -- 87 ) The Program Exchange ( 1987 -- 89 ) Columbia TriStar Television ( 1996 -- 2002 ) Sony Pictures Television ( 2002 -- present ) Release Original network ABC Audio format Monaural Original release September 25 , 1970 ( 1970 - 09 - 25 ) -- March 23 , 1974 ( 1974 - 03 - 23 ) Chronology Related shows Getting Together Goober and the Ghost Chasers Partridge Family 2200 A.D.
  Fact 0.0: The Partridge Family is a musical sitcom
  Fact 0.1: The Partridge Family was created by Bernard Slade
  Fact 0.2: The Partridge Family starred Shirley Jones
  Fact 0.3: The Partridge Family starred David Cassidy
  Fact 0.4: The Partridge Family starred Danny Bonaduce
  Fact 0.5: The Partridge Family starred Susan Dey
  Fact 0.6: The Partridge Family starred Suzanne Crough
  Fact 0.7: The Partridge Family starred Jeremy Gelbwaks
  Fact 0.8: The Partridge Family starred Dave Madden
  Fact 0.9: The Partridge Family starred Brian Forster
  Fact 0.10: The Partridge Family's theme music composers were Diane Hilderbrand, Danny Janssen and Wes Farrell
  Fact 0.11: The Partridge Family's opening theme was 'When We're Singin'' (1970-71)
  Fact 0.12: The Partridge Family's opening theme was 'C'mon, Get Happy' (1971-74)
  Fact 0.13: The Partridge Family's composers were George Duning, Benny Golson, Warren Barker, Hugo Montenegro and Shorty Rogers
  Fact 0.14: The Partridge Family is from the United States
  Fact 0.15: The Partridge Family's original language is English
  Fact 0.16: The Partridge Family has 96 episodes
  Fact 0.17: The Partridge Family's executive producer was Bob Claver
  Fact 0.18: The Partridge Family's producers were William Bickley, Paul Junger Witt, Dale McRaven, Larry Rosen and Mel Swope
  Fact 0.19: The Partridge Family's cinematography was by Fred Jackman, Jr. and Irving Lippman
  Fact 0.20: The Partridge Family uses a single-camera setup
  Fact 0.21: The Partridge Family has a running time of 25 minutes
  Fact 0.22: The Partridge Family was produced by Screen Gems Television (1970-1974)
  Fact 0.23: The Partridge Family was distributed by Columbia Pictures Television (1974-84; 1989-96)
  Fact 0.24: The Partridge Family was distributed by DFS Program Exchange (1984-87)
  Fact 0.25: The Partridge Family was distributed by The Program Exchange (1987-89)
  Fact 0.26: The Partridge Family was distributed by Columbia TriStar Television (1996-2002)
  Fact 0.27: The Partridge Family was distributed by Sony Pictures Television (2002-present)
  Fact 0.28: The Partridge Family aired originally on ABC
  Fact 0.29: The Partridge Family has a monaural audio format
  Fact 0.30: The Partridge Family originally aired from September 25, 1970 to March 23, 1974
  Fact 0.31: Related shows include Getting Together
  Fact 0.32: Related shows include Goober and the Ghost Chasers
  Fact 0.33: Related shows include Partridge Family 2200 A.D.

Retrieved items:
<item 1>
The Partridge Family
The Partridge Family - wikipedia The Partridge Family Jump to : navigation , search The Partridge Family Genre Musical sitcom Created by Bernard Slade Starring
Shirley Jones David Cassidy Danny Bonaduce Susan Dey Suzanne Crough Jeremy Gelbwaks Dave Madden Brian Forster Theme music composer Diane Hilderbrand Danny
Janssen Wes Farrell Opening theme `` When We 're Singin ' '' ( 1970 -- 71 ) `` C'mon , Get Happy '' ( 1971 -- 74 ) Composer ( s ) George Duning Benny Golson
Warren Barker Hugo Montenegro Shorty Rogers Country of origin United States Original language ( s ) English No. of seasons No. of episodes 96 ( list of episodes
) Production Executive producer ( s ) Bob Claver Producer ( s ) William Bickley Paul Junger Witt Dale McRaven Larry Rosen Mel Swope Cinematography Fred Jackman
, Jr .
</item 1>

<item 2>
The Partridge Family
Episodes ( edit ) Main article : List of The Partridge Family episodes Shirley Jones and Ricky Segall , season 4 Production ( edit ) At the end of the first
season , Jeremy Gelbwaks ' family moved out of the Los Angeles area , and the part of Chris was recast with actor Brian Forster . According to David Cassidy ,
Gelbwaks `` had a personality conflict with every person in the cast and the producers '' and especially did not get along with Cassidy or Bonaduce . A dog
named Simone was featured in the first season , but it was phased out during the second season . At the beginning of the fourth season , a four - year - old
neighbor , Ricky Stevens ( Ricky Segall ) , was featured and would sing a children 's song during each episode , but the character was dropped mid-season .
Music ( edit ) Music recorded for the pilot episode was produced by Monkees ' producer Shorty Rogers . Songs for the ongoing series were recorded by music
producer Wes Farrell . The Partridge Family songs heard in the show , and on the several Partridge Family singles and albums , were not performed as seen on -
screen .
</item 2>

<item 3>
The Partridge Family
The show featured the casts discussing the histories of their shows , although other than Jones and Fred MacMurray both portraying single parents of large
families , the two series had no narrative link . Reunion on Danny ! ( edit ) In 1995 , a majority of the cast appeared on Bonaduce 's talk show Danny ! ,
including Shirley Jones , Dave Madden , Jeremy Gelbwaks , Brian Forster , Suzanne Crough , Ricky Segall and the show 's executive producer Bob Claver . Susan
Dey was working on a movie at the time but called into the show to briefly reminisce with Bonaduce . David Cassidy was also unable to appear as he was working
on a new album at that time . The new Partridge Family ( edit ) In 2004 , VH1 produced a pilot for a syndicated The New Partridge Family , starring Suzanne Sole
as Shirley , Leland Grant as Keith , Emma Stone as Laurie and Spencer Tuskowski as Danny . The pilot was the only episode produced . DVD releases ( edit ) Sony
Pictures Home Entertainment has released all 4 Seasons of The Partridge Family in DVD Region 1 .
</item 3>

<item 4>
The Partridge Family
Actor David Cassidy sang lead on most of the records , and actor Shirley Jones can be heard on a few . But the other voices and all the instruments were
recorded by session musicians . The Partridge Family 's instrumental sound was provided by musicians informally known as The Wrecking Crew , who did many of the
pop hits of the day attributed to other performers . The actors pretended to play their instruments . None of the actors sang background ; they lip - synched to
recorded singing by The Love Generation , AKA The Ron Hicklin Singers , composed of brothers John and Tom Bahler , Jackie Ward ( who as Robin Ward had a # 14
hit `` Wonderful Summer '' on Dot Records in 1963 ) , and Ron Hicklin . Actor David Cassidy was originally to lip sync with the rest of the cast , but convinced
Farrell that he could sing and was allowed to join the studio ensemble as the lead singer . He and his step - mother , Shirley , are the only two show actors
allowed to sing for recording . The Partridge Family theme song , produced by Rogers , was shown over opening credits . More than one song was used as the theme
song .
</item 4>

<item 5>
The Partridge Family
The actors pretended while listening to recordings by session musicians who provided the real vocal and instrumental music attributed to The Partridge Family .
Notable guest stars ( edit ) During its four - season run , many actors made guest appearances on the show . Some of them were well known at the time , such as
Morey Amsterdam , John Astin , Carl Ballantine , John Banner , Edgar Buchanan , George Chakiris , Dick Clark ( who later hosted The Other Half from 2001 to 2003
with Danny Bonaduce ) , Jackie Coogan , Howard Cosell , Jodie Foster , Bernard Fox , Ned Glass , James Gregory , Margaret Hamilton , Pat Harrington Jr. , Arte
Johnson , Harvey Lembeck , Art Metrano , Mary Ann Mobley , Harry Morgan , Slim Pickens , Richard Pryor , Barbara Rhoades , William Schallert , Nita Talbot ,
Larry Wilcox , Dick Wilson , and William Windom . Others would later become famous in other roles , such as Meredith Baxter , Richard Bull , Bert Convy , Farrah
Fawcett , Norman Fell , Anthony Geary , Louis Gossett Jr.
</item 5>

<item 6>
The Partridge Family
L-R : Shirley Jones , Jeremy Gelbwaks , Suzanne Crough , Susan Dey , Danny Bonaduce and David Cassidy In the pilot episode , a group of musical siblings in the
fictitious city of San Pueblo , California ( said to be `` 40 miles from Napa County '' in episode 24 , A Partridge By Any Other Name ) convinces their widowed
mother , bank teller Shirley Partridge , to help them out by singing as they record a pop song in their garage . Through the efforts of precocious 10 - year -
old Danny they find a manager , Reuben Kincaid , who helps make the song a Top 40 hit . After more persuading , Shirley agrees that the family can go on tour .
They acquire an old school bus , a 1957 Chevrolet Series 6800 Superior , for touring , paint it with Mondrian - inspired patterns , and head to Las Vegas ,
Nevada for their first live gig at Caesars Palace . Subsequent episodes usually feature the band performing in various venues or in their garage . The shows
often contrast suburban life with the adventures of a show - business family on the road .
</item 6>

<item 7>
The Partridge Family
Tatum claimed that she had submitted a similar premise to Screen Gems prior to 1970 called Baker 's Half - Dozen . The matter was resolved out of court , with
Tatum receiving a reported $150,000 from Screen Gems . The Partridge Family , season 1 Cast and characters ( edit ) Shirley Jones as Shirley Partridge : vocals
, keyboard , tambourine , percussion David Cassidy as Keith Partridge : lead vocals , rhythm guitar , electric lead guitar , banjo Susan Dey as Laurie Partridge
: vocals , harmony , piano , Hammond organ , percussion Danny Bonaduce as Danny Partridge : vocals , bass guitar Jeremy Gelbwaks as Chris Partridge ( season 1 )
: vocals , drums Brian Forster as Chris Partridge ( seasons 2 -- 4 ) : vocals , drums Suzanne Crough as Tracy Partridge : tambourine , percussion Dave Madden as
Reuben Kincaid : band manager Ricky Segall as Ricky Stevens ( season 4 ) : singer Simone , the family 's pet dog ( season 1 only ) On the show 's soundtrack and
records , none of the actors actually played the indicated instruments , and only Jones and primarily Cassidy sang .
</item 7>

<item 8>
The Partridge Family
1 hit ( after The Chipmunks and The Archies ) . The song 's companion LP , The Partridge Family Album , reached Number 4 in the Billboard 200 . It was also
awarded a gold disc by the RIAA in December 1970 , having sold over 500,000 copies . A string of hit Partridge singles followed : `` Does n't Somebody Want to
Be Wanted '' , `` I 'll Meet You Halfway '' , `` I Woke Up In Love This Morning '' , `` It 's One of Those Nights ( Yes Love ) '' , `` Am I Losing You '' , ``
Looking Through the Eyes of Love '' , `` Breaking Up Is Hard to Do '' , and `` A Friend and a Lover '' . These singles were showcased on million - selling
albums including Up To Date , Sound Magazine , Shopping Bag , Notebook , Crossword Puzzle , and Bulletin Board . Their holiday album A Partridge Family
Christmas Card was the No. 1 selling Christmas record of 1971 .
</item 8>

<item 9>
The Partridge Family
, Harold Gould , Jackie Earle Haley , Mark Hamill , Season Hubley , Ann Jillian , Gordon Jump , Cheryl Ladd , Michael Lembeck , William Lucking , Stuart
Margolin , Richard Mulligan , Michael Ontkean , Noam Pitlik , Annette O'Toole , Charlotte Rae , Rob Reiner , Jack Riley , Jaclyn Smith , Vic Tayback , Nancy
Walker , and Frank Welker . The Partridge Family , season 3 Country singer Johnny Cash made an uncredited cameo appearance in the pilot episode . Ray Bolger
played Shirley 's father in three episodes , and Rosemary DeCamp played Shirley 's mother in four episodes . Then - Governor Ronald Reagan 's daughter , Maureen
Reagan , was also featured in one episode . Future Charlie 's Angels stars Jaclyn Smith , Farrah Fawcett and Cheryl Ladd all made guest appearances on separate
episodes . Bobby Sherman appeared in the last episode of the first season ( Episode 25 ) as struggling songwriter Bobby Conway . This episode led into a short -
lived spinoff series on ABC , Getting Together , starring Sherman and Wes Stern as Bobby Conway 's business partner Lionel Poindexter .
</item 9>

<item 10>
The Partridge Family
( hide ) The Partridge Family David Cassidy Shirley Jones Danny Bonaduce Susan Dey Jeremy Gelbwaks Brian Forster Suzanne Crough Dave Madden Studio albums The
Partridge Family Album Up to Date Sound Magazine A Partridge Family Christmas Card Shopping Bag The Partridge Family Notebook Crossword Puzzle Bulletin Board
Other albums At Home with Their Greatest Hits The World of the Partridge Family Greatest Hits The Definitive Collection Come On Get Happy ! : The Very Best of
The Partridge Family Singles `` I Think I Love You '' `` Does n't Somebody Want to Be Wanted '' `` I 'll Meet You Halfway '' `` I Woke Up in Love This Morning
'' `` It 's One of Those Nights ( Yes Love ) '' `` Breaking Up Is Hard to Do '' `` Looking Through the Eyes of Love '' Related articles List of The Partridge
Family episodes The Partridge Family discography David Cassidy discography Wes Farrell Goober and the Ghost Chasers Partridge Family 2200 A.D. MusicBrainz :
3376b487 - 95b4 - 42e6 - b7f4 - 71cffffcad97 Retrieved from `` https://en.wikipedia.org/w/index.
</item 10>
=== END ITEM id=196cc6e737ba17bb ===

=== ITEM id=c982ceba23d029c1 ===
Gold sentences and facts:
Sentence 0: Lynne is the eldest of the Slater sisters and she arrives in Albert Square in 2000 with her family : father Charlie ( Derek Martin ) ; sisters Kat ( Jessie Wallace ) , Little Mo ( Kacey Ainsworth ) and Zoe ( Michelle Ryan ) , and grandmother Mo ( Laila Morse ) . Unlike Little Mo and Zoe , Lynne is aware that Zoe is in fact Kat 's daughter and helps keep the family secret until Kat reveals the truth to Zoe .
  Fact 0.0: Lynne is the eldest of the Slater sisters.
  Fact 0.1: Lynne arrives in Albert Square in 2000.
  Fact 0.2: Lynne arrives in Albert Square with her family.
  Fact 0.3: Lynne's father is Charlie, played by Derek Martin.
  Fact 0.4: Lynne's sister Kat is played by Jessie Wallace.
  Fact 0.5: Lynne's sister Little Mo is played by Kacey Ainsworth.
  Fact 0.6: Lynne's sister Zoe is played by Michelle Ryan.
  Fact 0.7: Lynne's grandmother is Mo, played by Laila Morse.
  Fact 0.8: Unlike Little Mo and Zoe, Lynne is aware that Zoe is in fact Kat's daughter.
  Fact 0.9: Lynne helps keep the family secret that Zoe is Kat's daughter until Kat reveals the truth to Zoe.

Retrieved items:
<item 1>
Lynne Hobbs
Syed Masood Tamwar Masood Zainab Masood Lilly Mattock Josie McFarlane Poppy Meadow Darren Miller Demi Miller Keith Miller Mickey Miller Rosie Miller Rob Minter
Archie Mitchell Ben Mitchell Danny Mitchell Glenda Mitchell Grant Mitchell Jamie Mitchell Kate Mitchell Little Mo Mitchell Peggy Mitchell Ronnie Mitchell Roxy
Mitchell Sam Mitchell Tiffany Mitchell Vinnie Monks Alfie Moon Anthony Moon Danny Moon Eddie Moon Jake Moon Michael Moon Nana Moon Spencer Moon Tyler Moon
Trevor Morgan Rod Norman Andy O'Brien Mercy Olubunmi Ali Osman Mehmet Osman Sue Osman Steve Owen Annie Palmer George Palmer Lola Pearce Minty Peterson Frankie
Pierre Paul Priestly Irene Raymond Louise Raymond Simon Raymond Terry Raymond Dennis Rickman Binnie Roberts Roly Willy Roper Matthew Rose Colin Russell Mandy
Salter Trevor Short Lydia Simmonds Ethel Skinner Charlie Slater Sean Slater Zoe Slater Leon Small Babe Smith Gus Smith Mary Smith Dan Sullivan Gavin Sullivan
Dawn Swann Celestine and Etta Tavernier Clyde Tavernier Hattie Tavernier Jules Tavernier Kelly Taylor Heather Trott Anthony Trueman Paul Trueman Yolande Trueman
Owen Turner Angie Watts Chrissie Watts Den Watts Wellard Carl White Carly Wicks David Wicks Dean Wicks Joe
</item 1>

<item 2>
An extensive EastEnders character roster, including the Cotton, di Marco, Fowler, Jackson, Masood, Mitchell, Miller, Moon, Slater, Tavernier, Trueman, Watts and
Wicks families. It also lists numerous individual characters, from Lynne Hobbs, Dotty Cotton and Nick Cotton to Syed and Zainab Masood, Peggy and Grant
Mitchell, Alfie Moon, Kat and Zoe Slater, Den Watts and Wellard. Some names are repeated.
</item 2>

<item 3>
Lynne Hobbs (née Slater), played by Elaine Lordan, was an EastEnders regular from September 2000 to July 2004. A feisty, loud-mouthed café worker, she was part
of the Slater family introduced by producer John Yorke in 2000. Her relatives included father Charlie and sisters Kat, Little Mo and Belinda; she married Garry
Hobbs in 2001 and had daughter Vivienne. Lordan and producers mutually agreed Lyn
</item 3>

<item 4>
Lynne Hobbs
In January 2016 , Lynne attends Charlie 's funeral off - screen . Creation and development ( edit ) In 2000 , EastEnders ' executive producer John Yorke decided
to introduce the `` classic '' Slater family . He felt the show needed to go back to its roots and bring back some traditional values . BBC 's head of drama ,
Mal Young commented , `` We do not have enough solid families in the soap , there were a lot of fractured families and people who were alone . '' The family
were created as a replacement for the di Marco family , who were axed by Yorke . The family consisted of , grandmother Mo Harris ( Laila Morse ) , father
Charlie ( Derek Martin ) , his four children , Lynne ( Elaine Lordan ) , Kat ( Jessie Wallace ) , Little Mo ( Kacey Ainsworth ) , Zoe ( Michelle Ryan ) , as
well as Lynne 's boyfriend Garry Hobbs ( Ricky Groves ) . It later transpired that Kat was the mother of Zoe , after being raped by her uncle Harry Slater (
Michael Elphick ) .
</item 4>

<item 5>
Lynne Hobbs
links ( edit ) Lynne Hobbs at BBC Online EastEnders characters Present and future characters Kathy Beale Sharon Watts Ian Beale Dot Cotton Martin Fowler Phil
Mitchell Sonia Fowler Robbie Jackson Mel Owen Billy Mitchell Mo Harris Kat Slater Patrick Trueman Louise Mitchell Bobby Beale Stacey Slater Jean Slater Honey
Mitchell Denise Fox Max Branning Janet Mitchell Shirley Carter Jay Brown Rainie Cross Masood Ahmed Jack Branning Whitney Dean Tiffany Butcher Kim Fox Cora Cross
Tina Carter Linda Carter Mick Carter Charlie Cotton ( hide ) Past characters AJ Ahmed Della Alexander Johnny Allen Ruby Allen Aunt Sal Tom Banks Geoff Barnes
Clare Bates Debbie Bates Laurie Bates Nigel Bates Cindy Beale Jane Beale Laura Beale Lou Beale Lucy Beale Pete Beale Peter Beale Steven Beale Polly Becker Adam
Best Manda Best Abi Branning Alice Branning April Branning Bradley Branning Derek Branning Jim Branning Lauren Branning Suzy Branning Tanya Branning Aidan
Brosnan Diane Butcher Frank Butcher Janine Butcher Liam Butcher Mo Butcher Pat Butcher Ricky Butcher Kelvin Carpenter Tony Carpenter Nancy Carter Stan Carter
Zsa Zsa Carter Lorna Cartwright Barry Clark Christian Clarke Paul Coker Richard Cole Julie Cooper Ashley Cotton Charlie Cotton ( 1986
</item 5>

<item 6>
Lynne Hobbs
Lynne is a fierce judge of right and wrong , although she is not always able to practise what she preaches . She has a turbulent relationship with husband Garry
( Ricky Groves ) , but despite her complaints about him , it ends up being Lynne 's adultery that puts their relationship in jeopardy long before Garry strays .
Contents ( hide ) 1 Storylines 1.1 Backstory 1.2 2000 -- 2004 2 Creation and development 3 In popular culture 4 References 5 External links Storylines ( edit )
Backstory ( edit ) Lynne is the eldest of the Slater sisters and she arrives in Albert Square in 2000 with her family : father Charlie ( Derek Martin ) ;
sisters Kat ( Jessie Wallace ) , Little Mo ( Kacey Ainsworth ) and Zoe ( Michelle Ryan ) , and grandmother Mo ( Laila Morse ) . Unlike Little Mo and Zoe , Lynne
is aware that Zoe is in fact Kat 's daughter and helps keep the family secret until Kat reveals the truth to Zoe .
</item 6>

<item 7>
Lynne Hobbs, eldest of the Slater sisters, arrives in Albert Square in 2000 with her family and helps keep the secret that Zoe is Kat’s daughter. She wants
marriage and a family with her reluctant boyfriend, mechanic Garry, and eventually marries him. After clashing with co-worker Janine, Lynne becomes attracted to
Beppe and sleeps with him just before the wedding. In 2002, she rekindles things with ex-fiancé Jason, who
</item 7>

<item 8>
Lynne Hobbs
Chrissie Watts Den Watts Wellard Carl White Carly Wicks David Wicks Dean Wicks Joe Wicks Kevin Wicks Lorraine Wicks Simon Wicks Debbie Wilkins Cindy Williams
James Willmott - Brown Willy May Wright Lists of characters By year : 1985 1986 2017 2018 Others : Flaherty family Dickens Hill inmates The Banned band members
Characters from spin - offs Retrieved from `` https://en.wikipedia.org/w/index.php?title=Lynne_Hobbs&oldid=838104750 '' Categories : EastEnders characters
Fictional characters introduced in 2000 Female characters in television Hidden categories : Articles needing additional references from March 2011 All articles
needing additional references All articles with unsourced statements Articles with unsourced statements from January 2011 Use dmy dates from March 2011 Talk
Contents About Wikipedia Gaeilge Edit links This page was last edited on 24 April 2018 , at 23 : 33 . About Wikipedia
</item 8>

<item 9>
Lynne Hobbs
Unsourced material may be challenged and removed . ( March 2011 ) ( Learn how and when to remove this template message ) Lynne Hobbs EastEnders character
Portrayed by Elaine Lordan Duration 2000 -- 2004 First appearance Episode 2003 18 September 2000 ( 2000 - 09 - 18 ) Last appearance Episode 2756 2 July 2004 (
2004 - 07 - 02 ) Created by Tony Jordan Introduced by John Yorke Spin - off appearances Slaters in Detention ( 2003 ) Classification Former ; regular Profile
Other names Lynne Slater Occupation Café worker ( show ) Family Father Charlie Slater Mother Viv Slater Sisters Kat Slater Little Mo Mitchell Belinda Peacock
Husband Garry Hobbs ( 2001 -- 2004 ) Daughters Vivienne Hobbs Grandmothers Mo Harris Uncles Harry Slater Nephews Freddie Mitchell Nieces Zoe Slater Lynne Hobbs
( also Slater ) is a fictional character from the BBC soap opera EastEnders , played by Elaine Lordan from September 2000 to July 2004 .
</item 9>

<item 10>
List of The Fall characters
Katie is a gifted musician and uses online videos to express her feelings . During series 2 she becomes deeply embroiled in Paul 's life . Paul ties her up in a
hotel room , for example , whilst Katie breaks into a house wearing a shirt with Spector 's face on . Benedetto is placed under police surveillance and later
arrested whilst trying to dispose of evidence that implicates Paul in the murders . In series three , she attacks a girl with acid , and is detained for her
crimes . During this time , she reveals to Stella that her father 's death had a severe emotional impact on her , and she has been self - harming . Sally Ann
Spector ( edit ) Sally Ann Spector ( Bronagh Waugh ) , the wife of Paul Spector and a neonatal nurse . She has a deep devotion to Paul , but is wary of his
character as she has never met his family . Sally Ann is the mother of his two children and is unaware of his dark side . Over the course of the series , she
becomes embroiled in his crimes without her knowledge , and even provides a false alibi to the police .
</item 10>
=== END ITEM id=c982ceba23d029c1 ===
