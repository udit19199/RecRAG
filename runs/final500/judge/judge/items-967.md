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

=== ITEM id=9ea8a58cea1ec690 ===
Gold sentences and facts:
Sentence 0: The Homestead Acts were several United States federal laws that gave an applicant ownership of land , typically called a `` homestead , '' at no cost . In all , more than 270 million acres of public land , or nearly 10 % of the total area of the U.S. , was given away free to 1.6 million homesteaders ; most of the homesteads were west of the Mississippi River .
  Fact 0.0: The Homestead Acts were several United States federal laws.
  Fact 0.1: The Homestead Acts gave an applicant ownership of land.
  Fact 0.2: The land given under the Homestead Acts is typically called a homestead.
  Fact 0.3: The land was given at no cost.
  Fact 0.4: More than 270 million acres of public land were given away free under the Homestead Acts.
  Fact 0.5: The 270 million acres of public land were nearly 10% of the total area of the U.S.
  Fact 0.6: The public land was given away free to 1.6 million homesteaders.
  Fact 0.7: Most of the homesteads were west of the Mississippi River.
Sentence 1: An extension of the Homestead Principle in law , the Homestead Acts were an expression of the `` Free Soil '' policy of Northerners who wanted individual farmers to own and operate their own farms , as opposed to Southern slave - owners who wanted to buy up large tracts of land and use slave labor , thereby shutting out free white men .
  Fact 1.0: The Homestead Acts were an extension of the Homestead Principle in law.
  Fact 1.1: The Homestead Acts were an expression of the 'Free Soil' policy of Northerners.
  Fact 1.2: Northerners with the Free Soil policy wanted individual farmers to own and operate their own farms.
  Fact 1.3: Southern slave-owners, as opposed to Northerners, wanted to buy up large tracts of land.
  Fact 1.4: Southern slave-owners wanted to use slave labor on large tracts of land.
  Fact 1.5: Southern slave-owners' land purchases would shut out free white men.

Retrieved items:
<item 1>
Westward Expansion Trails
Westward expansion trails - wikipedia Westward expansion trails Jump to : navigation , search This article needs additional citations for verification . Please
help improve this article by adding citations to reliable sources . Unsourced material may be challenged and removed . ( March 2013 ) ( Learn how and when to
remove this template message ) In the American Old West , overland trails were popular means of travel used by pioneers and immigrants throughout the 19th
century and especially between 1830 and 1870 as an alternative to sea and railroad transport . These immigrants began to settle various regions of North America
west of the Great Plains as part of the mass overland migrations of the mid-19th century . Settlers emigrating from the eastern United States were spurred by
various motives , among them religious persecution , economic incentives , and the prospect of new wealth and new freedoms in what was then a largely unspoiled
and unregulated country . By the 1840s , several popular and well - established routes had been blazed through the continental interior to destinations in the
far west , including the Oregon Trail , California Trail , and Mormon Trail .
</item 1>

<item 2>
Homestead Acts
Some scholars believe the acreage limits were reasonable when the act was written , but reveal that no one understood the physical conditions of the plains .
According to Hugh Nibley , much of the rain forest west of Portland , Oregon was acquired by the Oregon Lumber Company by illegal claims under the Act .
Nonetheless , in 1995 , a random survey of 178 members of the Economic History Association found that 70 percent of economists and 84 percent of economic
historians disagreed that `` Nineteenth - century U.S. land policy , which attempted to give away free land , probably represented a net drain on the productive
capacity of the country . '' Related Acts in other countries ( edit ) Canada ( edit ) Similar laws were passed in Canada : The Legislative Assembly of Ontario
passed The Free Grants and Homestead Act in 1868 , which introduced a conditional scheme to an existing free grant plan previously authorized by the Province of
Canada in The Public Lands Act of 1860 .
</item 2>

<item 3>
Homestead Acts
History ( edit ) Donation land claim Act of 1850 ( edit ) Main article : Donation Land Claim Act The Donation Land Claim Act allowed settlers to claim land in
the Oregon Territory , then including the modern states of Washington , Oregon , Idaho and parts of Wyoming . Settlers were able to claim 320 or 640 acres of
land for free between 1850 and 1854 , and then at a cost of $1.25 per acres until the law expired in 1855 . Homestead Act of 1862 ( edit ) Certificate of
homestead in Nebraska given under the Homestead Act , 1862 Wikisource has original text related to this article : Homestead Act ( 1862 ) The `` yeoman farmer ''
ideal of Jeffersonian democracy was still a powerful influence in American politics during the 1840 -- 1850s , with many politicians believing a homestead act
would help increase the number of `` virtuous yeomen '' .
</item 3>

<item 4>
Westward Expansion Trails
From their labors arose the State of Deseret , later to become the Utah Territory , and finally the State of Utah . The Mormon pioneers shared similar
experiences with others traveling west : the drudgery of walking hundreds of miles , suffocating dust , violent thunderstorms , mud , temperature extremes , bad
water , poor forage , sickness , and death . They recorded their experiences in journals , diaries , and letters . The Mormons , however , were a unique part of
this migration . Their move to the Valley of the Great Salt Lake was a response to their violent expulsion from Missouri and Illinois . As it was also motivated
by a desire to maintain a religious and cultural identity it was necessary to find an isolated area where they could permanently settle and practice their
religion in peace . Southern Emigrant Trail ( edit ) Main article : Southern Emigrant Trail The Southern Emigrant Trail was a major land route for immigration
into California from the eastern United States that followed the Santa Fe Trail to New Mexico during the California Gold Rush .
</item 4>

<item 5>
Westward Expansion Trails
During the twenty - five years 1841 -- 1866 , 250,000 to 650,000 people `` pulled up stakes , '' and headed west along these trails . About one - third
immigrated to Oregon , one - third to California and one - third to Utah , Colorado , and Montana . Although it is often stated that the Northern trails began
in certain cities on the Missouri River , emigrants following any of the three trails typically left from one of three `` jumping off '' points on the Missouri
's steamboat serviced river ports : Independence , Missouri or Saint Joseph , Missouri , or Council Bluffs , Iowa . ( Once known as Kanesville , Iowa until 1852
; after river dredging in the early 1850s , the latter town at the Missouri - Platte confluence became the most common departure point since it was close in
proximity to the River Platte -- along which the eastern trails ascend to South Pass above Fort Laramie. ) The trails from these cities ( and several others )
converged in the mostly empty flatlands of central Nebraska near present - day Kearney , in the vicinity of Fort Kearney .
</item 5>

<item 6>
Homestead Acts
The federal government charged royalties for extraction of these resources from public lands . On the other hand , homesteading schemes were generally pointless
for land containing `` locatable minerals , '' such as gold and silver , which could be controlled through mining claims under the Mining Act of 1872 , for
which the federal government did not charge royalties . The government developed no systematic method to evaluate claims under the homestead acts . Land offices
relied on affidavits from witnesses that the claimant had lived on the land for the required period of time and made the required improvements . In practice ,
some of these witnesses were bribed or otherwise colluded with the claimant . Although not necessarily fraud , it was common practice for the eligible children
of a large family to claim nearby land as soon as possible . After a few generations , a family could build up a sizable estate . The homesteads were criticized
as too small for the environmental conditions on the Great Plains ; a homesteader using 19th - century animal - powered tilling and harvesting could not have
cultivated the 1500 acres later recommended for dry land farming .
</item 6>

<item 7>
Westward Expansion Trails
Regardless of the trail used , the journey was often slow and arduous , fraught with risks from infectious diseases , dehydration , malnutrition , injury , and
harsh weather , with as many as one in ten travelers dying along the way , usually as a result of disease . The history of these trails and the pioneers who
traveled them have since become deeply embedded in the culture and folklore of the United States as some of the most significant influences to shape the content
and character of the nation . The remains of many trail ruts can still be observed in various locations throughout the American West . Travelers may loosely
follow various routes of the emigrant trails on modern highways through the use of byway signs across the western states . Contents ( hide ) 1 Trail choices 2
Santa Fe Trail 3 Old Spanish Trail 4 Oregon Trail 5 California Trail 6 Mormon Trail 7 Southern Emigrant Trail 8 See also 9 References Trail choices ( edit )
Travelers across what became the Western United States in the 19th century had the choice of several routes . Some of the earliest were those of the Mexicans in
the southwest . American trade with Northern Mexico created the Santa Fe Trail between St.
</item 7>

<item 8>
Homestead Acts
The Free Soil Party of 1848 -- 52 , and the new Republican Party after 1854 , demanded that the new lands opening up in the west be made available to
independent farmers , rather than wealthy planters who would develop it with the use of slaves forcing the yeomen farmers onto marginal lands . Southern
Democrats had continually fought ( and defeated ) previous homestead law proposals , as they feared free land would attract European immigrants and poor
Southern whites to the west . After the South seceded and their delegates left Congress in 1861 , the Republicans and other supporters from the upper South
passed a homestead act . The intent of the first Homestead Act , passed in 1862 , was to liberalize the homesteading requirements of the Preemption Act of 1841
. Its leading advocates were Andrew Johnson , George Henry Evans and Horace Greeley . The homestead was an area of public land in the West ( usually 160 acres
or 65 ha ) granted to any US citizen willing to settle on and farm the land for at least five years .
</item 8>

<item 9>
Westward Expansion Trails
After the end of the Mexican -- American War in 1849 , vast new American conquests again enticed mass immigration . Legislation like the Donation Land Claim Act
and significant events like the California Gold Rush further lured colonists to travel overland to the west . Two major wagon - based transportation networks ,
one typically starting in Missouri and the other in the Mexican province of Santa Fe de Nuevo México , served the majority of migrants during the era of
westward expansion . Three of the Missouri - based routes -- the Oregon , Mormon , and California Trails -- were collectively known as the Emigrant Trails .
Historians have estimated at least 500,000 emigrants used these three trails between 1843 and 1869 , and despite growing competition from transcontinental
railroads , some use even continued into the early 20th century . The major southern routes were the Santa Fe Trail , the Southern Emigrant Trail , and the Old
Spanish Trail , as well as its wagon road successor the Mormon Road , a southern spur of the California Trail used in the winter that also made use of the
western half of the Old Spanish Trail .
</item 9>

<item 10>
Homestead Acts
Homestead Acts - wikipedia Homestead Acts Jump to : navigation , search Not to be confused with Homesteading . Norwegian settlers in 1898 North Dakota in front
of their homestead , a sod hut The Homestead Acts were several United States federal laws that gave an applicant ownership of land , typically called a ``
homestead , '' at no cost . In all , more than 270 million acres of public land , or nearly 10 % of the total area of the U.S. , was given away free to 1.6
million homesteaders ; most of the homesteads were west of the Mississippi River . An extension of the Homestead Principle in law , the Homestead Acts were an
expression of the `` Free Soil '' policy of Northerners who wanted individual farmers to own and operate their own farms , as opposed to Southern slave - owners
who wanted to buy up large tracts of land and use slave labor , thereby shutting out free white men . The first of the acts , the Homestead Act of 1862 , opened
up millions of acres . Any adult who had never taken up arms against the U.S. government could apply .
</item 10>
=== END ITEM id=9ea8a58cea1ec690 ===

=== ITEM id=359f45d5f3c31fb6 ===
Gold sentences and facts:
Sentence 0: The new season of The Little Couple premiered on September 19 , 2017 .
  Fact 0.0: A new season of The Little Couple premiered on September 19, 2017.

Retrieved items:
<item 1>
The Little Couple
2012 ) Anderson ( March 20 , 2012 ) Episodes ( edit ) Season Episodes Originally aired First aired Last aired 14 May 26 , 2009 August 4 , 2009 18 October 27 ,
2009 February 23 , 2010 19 June 1 , 2010 December 7 , 2010 24 May 31 , 2011 November 1 , 2011 5 35 March 20 , 2012 October 15 , 2013 6 16 March 4 , 2014 June 17
, 2014 7 12 December 2 , 2014 June 30 , 2015 8 10 January 5 , 2016 March 8 , 2016 Season 1 ( 2009 ) ( edit ) No. in series No. in season Title Original air date
0 0 `` Little People : Just Married '' January 19 , 2009 ( 2009 - 01 - 19 ) Bill Klein and Jennifer Arnold are introduced in this one - hour special following
them before , during , and after their wedding .
</item 1>

<item 2>
The Little Couple
Arnold is a neonatologist who works at the Texas Children 's Hospital , and Klein is a medical supplies and telemarketing businessman . They were originally
introduced in a one - hour TLC special , Little People : Just Married . Season 1 of the show premiered on May 26 , 2009 . Season 2 premiered on October 27 ,
2009 . On February 22 , 2010 , TLC renewed The Little Couple for a third season consisting of 20 episodes . The mid-season finale aired after 9 episodes on July
27 , 2010 , and season 3 returned on October 12 , 2010 . It was announced on Facebook that the show would be returning for its 4th season on May 31 , 2011 .
Again , the season was split , with the second half of season 4 premiering on September 27 , 2011 . In February 2012 , Klein opened Rocky&Maggie in Rice Village
Houston . named after their pets Chihuahua Rocky and rescue mutt terrier Maggie . Arnold 's mother Judy manages the store .
</item 2>

<item 3>
The Little Couple
The Little Couple - wikipedia The Little Couple Jump to : navigation , search The Little Couple Genre Reality Starring Bill Klein Jennifer Arnold - Klein
William Rijin Klein Zoey Nidhi Klein Country of origin United States Original language ( s ) English No. of seasons 8 No. of episodes 134 Production Executive
producer ( s ) Eric Schotz Ruth Rivin Melissa Tallerine Producer ( s ) Elizabeth Massie Oliver Oertel Sonya Gay Bourn Editor ( s ) Adrian Herrera Paul Yates
Ryan Ely Scott Reynolds Camera setup Multiple Running time 22 to 24 minutes Production company ( s ) LMNO Productions Release Original network TLC Picture
format 480i ( SDTV ) 1080i ( HDTV ) Original release May 26 , 2009 ( 2009 - 05 - 26 ) -- present External links Website www.thelittlecouple.net The Little
Couple is an American reality television series on TLC that debuted May 26 , 2009 . The series features Bill Klein , a businessman , and Dr. Jennifer Arnold , a
neonatologist , who both have skeletal dysplasia .
</item 3>

<item 4>
The Little Couple
14 65 `` Will You Be My Surrogate ? '' September 27 , 2011 ( 2011 - 09 - 27 ) 1.31 The Little Couple meet Cindy and James , the couple that may be their
surrogate family . 15 66 `` See You Later , Alligator ? '' September 27 , 2011 ( 2011 - 09 - 27 ) 1.01 While Bill goes camping , Jen works with a local interior
designer . 16 67 `` Uncle Bill '' October 4 , 2011 ( 2011 - 10 - 04 ) 1.09 The couple heads to Florida to meet the newest member of the Klein family . The baby
reinforces Bill and Jen 's desire that they are ready to become parents . 17 68 `` Mother Knows Best '' October 4 , 2011 ( 2011 - 10 - 04 ) 1.04 Jen 's mom is
in town to help the couple pack up their rental house . The interior designer shows them his design plans for the new house .
</item 4>

<item 5>
The Little Couple
15 '' . Showbuzz Daily . The Little Couple Starring Bill Klein Jennifer Arnold Episodes Little Couple : Just Married TLC original programming 7 Little Johnstons
10 Years Younger 48 Hours 90 Day Fiancé All About Sex Breaking Amish Buying Naked Cake Boss Curvy Brides Extreme Cheapskates Four Weddings Gypsy Sisters A
Haunting Home Sweet Bus I Am Jazz I Found The Gown Leah Remini : It 's All Relative The Little Couple Little People , Big World Long Island Medium Long Lost
Family Love , Lust or Run My 600 - lb Life My Big Fat Fabulous Life My Big Fat Gypsy Wedding My Big Fat American Gypsy Wedding My Giant Life My Strange
Addiction Next Great Baker Kate Plus 8 Kindred Spirits OutDaughtered Outrageous 911 Our Little Family Paranormal Lockdown Randy to the Rescue Say Yes to the
Dress Say Yes to the Dress : Atlanta Say Yes to the Dress : Bridesmaids Say Yes to the Dress : Randy Knows Best Say Yes to the Dress : The Big Day Secretly
Pregnant Sex Sent Me to the ER Something Borrowed , Something New The Willis Family Too Close to Home Untold Stories of the E.R.
</item 5>

<item 6>
The Little Couple
18 50 `` Pushing the House Forward '' November 30 , 2010 ( 2010 - 11 - 30 ) 1.59 Jen is needed on a case at the hospital leaving Bill to communicate her visions
for the look of the house . 19 51 `` This Changes Everything '' December 7 , 2010 ( 2010 - 12 - 07 ) N / A A big week for the couple ; Bill 's parents are in
town , and a big conversation with Dr. Boostanfar is on the horizon . The couple learn that Jen 's hormone shots are not doing as well as they first expected .
Season 4 ( 2011 ) ( edit ) No. in series No. in season Title Original air date U.S. viewers ( millions ) 52 `` Third Time 's a Charm ? '' May 31 , 2011 ( 2011 -
05 - 31 ) 1.19 The discovery of two fertile eggs leave Bill and Jen to wonder if there will be a reason to celebrate .
</item 6>

<item 7>
The Little Couple
`` ' The Little Couple ' Season 6 to Premiere Tuesday , March 4 on TLC '' . TV by the Numbers ( Press release ) . Retrieved February 5 , 2014 . Jump up ^ ``
Tuesday 's Cable Ratings : 10 - 19 - 10 '' . The Futon Critic . Jump up ^ `` Tuesday 's Cable Ratings : 10 - 26 - 10 '' . The Futon Critic . Jump up ^ ``
Tuesday 's Cable Ratings : 11 - 2 - 10 '' . The Futon Critic . Jump up ^ `` Tuesday 's Cable Ratings : 11 - 9 - 10 '' . The Futon Critic . Jump up ^ `` Tuesday
's Cable Ratings : 11 - 23 - 10 '' . The Futon Critic . Jump up ^ `` Tuesday 's Cable Ratings : 11 - 30 - 10 '' . The Futon Critic . ^ Jump up to : `` Tuesday
's Cable Ratings : 5 - 31 - 11 '' . The Futon Critic .
</item 7>

<item 8>
The Little Couple
22 73 `` The Start of Something Big '' October 25 , 2011 ( 2011 - 10 - 25 ) 1.12 Bill and Jen ' must decide if they are willing to have their two embryos go
through genetic testing . 23 74 `` A Little More Q & A '' October 25 , 2011 ( 2011 - 10 - 25 ) 1.14 Bill and Jen take viewer questions and an inside view of the
past four seasons . 24 75 `` Are We Pregnant ? '' November 1 , 2011 ( 2011 - 11 - 01 ) 1.50 Bill and Jen 's embryos have thawed and now it 's time to transfer
them to their surrogate . Season 5 ( 2012 -- 13 ) ( edit ) No. in series No. in season Title Original air date U.S. viewers ( millions ) 76 `` Getting Back on
Track '' March 20 , 2012 ( 2012 - 03 - 20 ) 1.11 Bill and Jen are ready to move forward following their surrogate 's miscarriage .
</item 8>

<item 9>
The Little Couple
The new season of The Little Couple premiered on September 19 , 2017 . Television appearances ( edit ) The Oprah Winfrey Show ( March 9 , 2009 ) The Today Show
( July 7 , 2009 ) The Dr. Oz Show ( October 1 , 2009 ) Good Day L.A. ( December 16 , 2009 ) The Wendy Williams Show ( May 25 , 2010 ) ABC News ( May 25 , 2010 )
KHOU Great Day Houston ( August 19 , 2010 ) The Doctors ( 2008 TV series ) ( October 6 , 2010 ) ABC 13 ( November 18 , 2010 ) Good Day L.A. ( August 2 , 2011 )
KHOU : Great Day Houston ( August 2 , 2011 ) Extreme Makeover : Home Edition ( October 16 , 2011 ) KHOU : Great Day Houston ( November 23 , 2011 ) KHOU : Great
Day Houston ( February 2 , 2012 ) Fox News ( March 19 , 2012 ) Anderson ( March 20 ,
</item 9>

<item 10>
The Little Couple
9 135 `` New Challenges '' March 1 , 2016 ( 2016 - 03 - 01 ) N / A Jen and Bill have a night out just for adults at a local escape room . Meanwhile , Zoey has
her first Indian dance lesson ; and the family attends a Chinese lantern festival in Houston . 10 136 `` A Little 4 - 1 - 1 '' March 8 , 2016 ( 2016 - 03 - 08 )
N / A Bill and Jen answer questions and go over topics like parenting , issues from being little people , and having their own television show . References (
edit ) Jump up ^ `` TLC Heats Up Summer Schedule with New Premieres '' . The Futon Critic. April 27 , 2009 . Retrieved April 19 , 2013 . Jump up ^ Srikameswaran
, Anita ( August 7 , 2001 ) . `` Meet Dr. Arnold '' . Post-Gazette . Retrieved August 6 , 2013 . Jump up ^ George , Cindy ( May 24 , 2009 ) .
</item 10>
=== END ITEM id=359f45d5f3c31fb6 ===

=== ITEM id=7267647095f83526 ===
Gold sentences and facts:
Sentence 0: His roles continued in the 1990s with the TV series Cracker ( 1993 -- 1996 , returning in 2006 for a one - off special ) , in which he starred as forensic psychologist Dr. Edward `` Fitz '' Fitzgerald . The role won him three BAFTA awards . Roles in bigger films followed ; the James Bond films GoldenEye ( 1995 ) and The World Is Not Enough ( 1999 ) , a supporting role in From Hell ( 2001 ) , as well as half - giant Rubeus Hagrid in the Harry Potter films ( 2001 -- 2011 ) . J.K. Rowling , author of the Harry Potter series , had Coltrane at the top of her list to play Hagrid and , when asked whom she would like to see in the role , responded `` Robbie Coltrane for Hagrid '' in one quick breath . There is also an unnamed 8ft actor who stands in for the 6ft 1in Coltrane in some scenes .
  Fact 0.0: His roles continued in the 1990s.
  Fact 0.1: He appeared in the TV series Cracker from 1993 to 1996.
  Fact 0.2: He returned to the TV series Cracker in 2006 for a one-off special.
  Fact 0.3: In Cracker he starred as forensic psychologist Dr. Edward "Fitz" Fitzgerald.
  Fact 0.4: The role in Cracker won him three BAFTA awards.
  Fact 0.5: Roles in bigger films followed.
  Fact 0.6: He appeared in the James Bond film GoldenEye (1995).
  Fact 0.7: He appeared in the James Bond film The World Is Not Enough (1999).
  Fact 0.8: He had a supporting role in From Hell (2001).
  Fact 0.9: He played half-giant Rubeus Hagrid in the Harry Potter films from 2001 to 2011.
  Fact 0.10: J.K. Rowling, author of the Harry Potter series, had him at the top of her list to play Hagrid.
  Fact 0.11: When asked whom she would like to see in the role, Rowling responded "Robbie Coltrane for Hagrid" in one quick breath.
  Fact 0.12: An unnamed 8ft actor stands in for the 6ft 1in Coltrane in some scenes.
Sentence 1: Robbie Coltrane OBE Coltrane in April 2007 Anthony Robert McMillan ( 1950 - 03 - 30 ) 30 March 1950 ( age 67 ) Rutherglen , South Lanarkshire , Scotland Nationality British Alma mater University of Edinburgh Occupation Actor , Author Years active 1979 -- present Height 6 ft 1 in ( 185 cm ) Spouse ( s ) Rhona Gemmell ( m . 1999 -- 2003 ) Children
  Fact 1.0: Robbie Coltrane holds the OBE honour.
  Fact 1.1: A photo caption shows Coltrane in April 2007.
  Fact 1.2: Coltrane's birth name is Anthony Robert McMillan.
  Fact 1.3: Coltrane was born on 30 March 1950.
  Fact 1.4: Coltrane is listed as age 67.
  Fact 1.5: Coltrane was born in Rutherglen, South Lanarkshire, Scotland.
  Fact 1.6: Coltrane's nationality is British.
  Fact 1.7: Coltrane's alma mater is the University of Edinburgh.
  Fact 1.8: Coltrane's occupations are actor and author.
  Fact 1.9: Coltrane has been active as a performer from 1979 to the present.
  Fact 1.10: Coltrane's height is 6 ft 1 in (185 cm).
  Fact 1.11: Coltrane's spouse was Rhona Gemmell, married from 1999 to 2003.
  Fact 1.12: The children field is listed with no value.

Retrieved items:
<item 1>
Harry Potter (film series)
Warner Bros. had initially planned to release the first film over the 4 July 2001 weekend , making for such a short production window that several of the
originally proposed directors had withdrawn themselves from contention . Eventually , due to time constraints , the date was put back to 16 November 2001 .
Casting the roles of Harry , Ron , and Hermione Daniel Radcliffe , Emma Watson , and Rupert Grint at the world premiere of Harry Potter and the Deathly Hallows
-- Part 2 in Trafalgar Square , London on 7 July 2011 . In 2000 , after a seven - month search , lead actor Daniel Radcliffe was discovered by producer David
Heyman and writer Steve Kloves seated just behind them in a theatre . In Heyman 's own words , `` There sitting behind me was this boy with these big blue eyes
. It was Dan Radcliffe . I remember my first impressions : He was curious and funny and so energetic . There was real generosity too , and sweetness . But at
the same time he was really voracious and with hunger for knowledge of whatever kind . ''
</item 1>

<item 2>
Harry Potter (film series)
Retrieved 8 July 2007 . Jump up ^ Brian Linder ( 17 May 2000 ) . `` Bewitched Warner Bros. Delays Potter '' . IGN . Retrieved 8 July 2007 . Jump up ^ ``
Ultimate Edition : Screen Test , Trio Casting and Finding Harry Potter '' . Mugglenet.com . Archived from the original on 23 June 2011 . Retrieved 19 October
2017 . ^ Jump up to : `` Young Daniel gets Potter part '' . BBC Online . 21 August 2000 . Jump up ^ `` Daniel Radcliffe , Rupert Grint and Emma Watson bring
Harry , Ron and Hermione to life for Warner Bros. Pictures ' `` Harry Potter and the Sorcerer 's Stone '' `` . Warner Brothers . 21 August 2000 . Archived from
the original on 14 April 2002 . Retrieved 26 August 2010 . Jump up ^ `` Warner Bros. Studio Tour London - The Making of Harry Potter '' . www.wbstudiotour.co.uk
. Jump up ^ `` Warner Bros.
</item 2>

<item 3>
Robbie Coltrane
Hyde voice Van Helsing Mr. Hyde voice 2005 Harry Potter and the Goblet of Fire Rubeus Hagrid 2006 Stormbreaker The Prime Minister of Europe 2006 Provoked Lord
Edward Foster 2007 Harry Potter and the Order of the Phoenix Rubeus Hagrid 2008 Tale of Despereaux , The The Tale of Despereaux Gregory voice 2008 Brothers
Bloom , The The Brothers Bloom The Curator 2009 Harry Potter and the Half - Blood Prince Rubeus Hagrid 2009 Gooby Gooby voice Harry Potter and the Deathly
Hallows -- Part 1 Rubeus Hagrid 2011 Harry Potter and the Deathly Hallows -- Part 2 Rubeus Hagrid 2012 Brave Lord Dingwall voice Great Expectations Mr.
</item 3>

<item 4>
Harry Potter (film series)
Retrieved 8 July 2007 . Jump up ^ Davidson , Paul ( 15 March 2000 ) . `` Harry Potter Director Narrowed Down '' . IGN . Archived from the original on 6 December
2008 . Retrieved 8 July 2007 . Jump up ^ `` Terry Gilliam bitter about Potter '' . Wizard News . 29 August 2005 . Retrieved 8 July 2007 . ^ Jump up to : Linder
, Bran ( 28 March 2000 ) . `` Chris Columbus to Direct Harry Potter '' . IGN . Archived from the original on 13 January 2008 . Retrieved 8 July 2007 . Jump up ^
Sragow , Michael ( 24 February 2000 ) . `` A Wizard of Hollywood '' . Salon . Retrieved 8 July 2007 . Jump up ^ Linder , Brian ( 30 March 2000 ) . `` Chris
Columbus Talks Potter '' . IGN . Archived from the original on 6 December 2008 . Retrieved 8 July 2007 .
</item 4>

<item 5>
Harry Potter (film series)
My favourite character is Hermione . ' And I think for some weird reason , from that moment on , she sort of trusted me . '' Cast and crew See also : List of
Harry Potter cast members Aside from the three lead actors , other notable cast members include Robbie Coltrane as Rubeus Hagrid , Tom Felton as Draco Malfoy ,
Alan Rickman as Severus Snape , and Dame Maggie Smith as Minerva McGonagall . Richard Harris , who played the role of Professor Albus Dumbledore , died on 25
October 2002 causing the role to be re-cast for the third instalment , Harry Potter and the Prisoner of Azkaban . David Heyman and director Alfonso Cuarón chose
Michael Gambon to portray the character of Dumbledore , which he did for all succeeding films .
</item 5>

<item 6>
Harry Potter (film series)
Notable recurring cast members include Helena Bonham Carter as Bellatrix Lestrange , Warwick Davis as Filius Flitwick , Ralph Fiennes as Lord Voldemort ,
Brendan Gleeson as Alastor Moody , Richard Griffiths as Vernon Dursley , Jason Isaacs as Lucius Malfoy , Gary Oldman as Sirius Black , Fiona Shaw as Petunia
Dursley , Timothy Spall as Peter Pettigrew , David Thewlis as Remus Lupin , Emma Thompson as Sybill Trelawney , Mark Williams as Arthur Weasley , and Julie
Walters as Molly Weasley . The series has seen many returning crew members from various departments , including Tim Burke , visual effects supervisor ; Peter
Doyle , digital film colourist ; Nick Dudman , make - up and creature effects designer ; David Holmes , stunt double ; Amanda Knight , make - up artist ;
Stephenie McMillan , set designer ; Greg Powell , stunt coordinator ; Jany Temime , costume designer ; and Fiona Weir , casting director . Set design The Hall
of Christ Church in Oxford , England , the inspiration for the studio film set of The Great Hall of Hogwarts . The production designer for all eight films is
Stuart Craig .
</item 6>

<item 7>
Harry Potter (film series)
2001 ) The Chamber of Secrets ( 2002 ) The Prisoner of Azkaban ( 2004 ) The Goblet of Fire ( 2005 ) The Order of the Phoenix ( 2007 ) The Half - Blood Prince (
2009 ) The Deathly Hallows -- Part 1 ( 2010 ) The Deathly Hallows -- Part 2 ( 2011 ) Music The Philosopher 's Stone The Chamber of Secrets The Prisoner of
Azkaban The Goblet of Fire The Order of the Phoenix The Half - Blood Prince The Deathly Hallows -- Part 1 The Deathly Hallows -- Part 2 Related Cast members
Production of The Deathly Hallows Characters Main Harry Potter Ron Weasley Hermione Granger Lord Voldemort Albus Dumbledore Severus Snape Rubeus Hagrid Draco
Malfoy Supporting Ginny Weasley Fred and George Weasley Hogwarts staff Order of the Phoenix Dumbledore 's Army Death Eaters Fictional universe Hogwarts Magic
Magical creatures Magical objects Ministry of Magic Muggle Places Potions Quidditch Related works Fantastic Beasts and Where to Find Them Quidditch Through the
Ages The Tales of Beedle the Bard Prequel Pottermore The Cursed Child Hogwarts
</item 7>

<item 8>
Harry Potter (film series)
Radcliffe had already established himself as an actor in the 1999 BBC television production of David Copperfield in which he played the title role 's childhood
years . Heyman persuaded Radcliffe 's parents to allow him to audition for the part of Harry Potter , which involved Radcliffe being filmed . ( This screen test
footage was released via the first set of Ultimate Editions in 2009 . ) Rowling was enthusiastic after viewing Radcliffe 's filmed test , saying she did n't
think there was a better choice for the part of Harry Potter . Also in 2000 , the then unknown British actors Emma Watson and Rupert Grint were selected from
thousands of auditioning children to play the roles of Hermione Granger and Ron Weasley , respectively . Their only previous acting experience was in school
plays . Grint was eleven years old and Watson ten at the time they were cast . Los Angeles Times writer Geoff Boucher , who conducted the above - mentioned
interview with Heyman , added that the casting of the three major roles `` is especially impressive in hindsight .
</item 8>

<item 9>
Robbie Coltrane
J.K. Rowling , author of the Harry Potter series , had Coltrane at the top of her list to play Hagrid and , when asked whom she would like to see in the role ,
responded `` Robbie Coltrane for Hagrid '' in one quick breath . There is also an unnamed 8ft actor who stands in for the 6ft 1in Coltrane in some scenes .
Coltrane has also presented a number of documentary programmes for the British ITV network based around his twin passions for travel and transportation .
Coltrane in a Cadillac ( 1993 ) saw him cross North America from Los Angeles to New York behind the wheel of a 1951 Cadillac Series 62 coupe convertible , a
journey of 3,765 miles ( 6,059 km ) which he completed in 32 days . In 1997 , Coltrane appeared in a series of six programmes under the title Coltrane 's Planes
and Automobiles , in which he extolled the virtues of the steam engine , the diesel engine , the supercharger , the V8 engine , the two - stroke engine , and
the jet engine .
</item 9>

<item 10>
Robbie Coltrane
Retrieved 15 August 2011 . ( Accio Quote ! , the Largest Archive of J.K. Rowling quotes on the web ) Jump up ^ Ezard , John ( 5 June 2002 ) . `` Size means
everything as Harry 's big secret slips out '' . The Guardian . Retrieved 1 October 2016 . Jump up ^ Femia , Will ( 13 November 2001 ) . `` Chat Flashback : The
cinematic debut of Rubeus Hagrid '' . MSNBC . NBC News . Retrieved 1 October 2016 . In real life , I 'm 6'1 '' -- each way pretty well , north and south and
east and west , unfortunately Jump up ^ The Robbie Coltrane Special at the Internet Movie Database Jump up ^ `` The Gruffalo 's Child '' . BBC One . Retrieved
10 December 2011 . External links ( edit ) Wikimedia Commons has media related to Robbie Coltrane .
</item 10>
=== END ITEM id=7267647095f83526 ===

=== ITEM id=318df246095ee282 ===
Gold sentences and facts:
Sentence 0: In 1910 , in conjunction with advocacy work led by Florence Kelley of the National Consumer League , the Women 's Trade Union League ( WTLU ) of Massachusetts under the leadership of Elizabeth Evans took up the cause of minimum wage legislation in Massachusetts . Over the next two years , a coalition of social reform groups and labor advocates in Boston pushed for minimum wage legislation in the state . In 1912 , Massachusetts passed the first American minimum wage legislation , which established a state commission for recommending non-compulsory minimum wages for women and children . Within eight years , at least thirteen U.S. states and the District of Columbia would pass minimum wage laws , with pressure being placed on state legislatures by the National Consumers League in a coalition with other women 's voluntary associations and organized labor . The United States Supreme Court of the Lochner era consistently invalidated compulsory minimum wage laws . Advocates for these minimum wage laws hoped that they would be upheld under the precedent of Muller v. Oregon , which upheld maximum working hours laws for women on the grounds that women required special protection that men did not . However , the Court did not extend this principle to minimum wage laws , considering the latter as interfering with the ability of employers to freely negotiate wage contracts with employees .
  Fact 0.0: In 1910, the Women's Trade Union League (WTLU) of Massachusetts, under the leadership of Elizabeth Evans, took up the cause of minimum wage legislation in Massachusetts.
  Fact 0.1: The 1910 effort was in conjunction with advocacy work led by Florence Kelley of the National Consumer League.
  Fact 0.2: Over the next two years, a coalition of social reform groups and labor advocates in Boston pushed for minimum wage legislation in the state.
  Fact 0.3: In 1912, Massachusetts passed the first American minimum wage legislation.
  Fact 0.4: The 1912 Massachusetts law established a state commission for recommending non-compulsory minimum wages for women and children.
  Fact 0.5: Within eight years, at least thirteen U.S. states and the District of Columbia would pass minimum wage laws.
  Fact 0.6: Pressure on state legislatures came from the National Consumers League, in a coalition with other women's voluntary associations and organized labor.
  Fact 0.7: The United States Supreme Court of the Lochner era consistently invalidated compulsory minimum wage laws.
  Fact 0.8: Advocates for minimum wage laws hoped they would be upheld under the precedent of Muller v. Oregon.
  Fact 0.9: Muller v. Oregon upheld maximum working hours laws for women on the grounds that women required special protection that men did not.
  Fact 0.10: The Court did not extend the Muller v. Oregon principle to minimum wage laws.
  Fact 0.11: The Court considered minimum wage laws as interfering with the ability of employers to freely negotiate wage contracts with employees.
Sentence 1: In 1933 , the Roosevelt administration made the first attempt at establishing a national minimum wage , when a $0.25 per hour standard was set as part of the National Industrial Recovery Act . However , in the 1935 court case Schechter Poultry Corp. v. United States ( 295 U.S. 495 ) , the US Supreme Court declared the act unconstitutional , and the minimum wage was abolished . In 1938 , the minimum wage was re-established pursuant to the Fair Labor Standards Act , once again at $0.25 per hour ( $4.78 in 2017 dollars ) . In 1941 , the Supreme Court upheld the Fair Labor Standards Act in United States v. Darby Lumber Co. , holding that Congress had the power under the Commerce Clause to regulate employment conditions .
  Fact 1.0: In 1933, the Roosevelt administration made the first attempt at establishing a national minimum wage.
  Fact 1.1: The 1933 minimum wage attempt set a $0.25 per hour standard as part of the National Industrial Recovery Act.
  Fact 1.2: In the 1935 court case Schechter Poultry Corp. v. United States (295 U.S. 495), the US Supreme Court declared the National Industrial Recovery Act unconstitutional.
  Fact 1.3: The national minimum wage was abolished after the Schechter Poultry ruling.
  Fact 1.4: In 1938, the minimum wage was re-established under the Fair Labor Standards Act.
  Fact 1.5: The 1938 minimum wage was $0.25 per hour.
  Fact 1.6: The 1938 rate of $0.25 per hour equals $4.78 in 2017 dollars.
  Fact 1.7: In 1941, the Supreme Court upheld the Fair Labor Standards Act in United States v. Darby Lumber Co.
  Fact 1.8: In United States v. Darby Lumber Co., the Court held that Congress had the power under the Commerce Clause to regulate employment conditions.

Retrieved items:
<item 1>
Minimum wage in the United States
, holding that Congress had the power under the Commerce Clause to regulate employment conditions . The 1938 minimum wage law only applied to `` employees
engaged in interstate commerce or in the production of goods for interstate commerce , '' but in amendments in 1961 and 1966 , the federal minimum wage was
extended ( with slightly different rates ) to employees in large retail and service enterprises , local transportation and construction , state and local
government employees , as well as other smaller expansions ; a grandfather clause in 1990 drew most employees into the purview of federal minimum wage policy ,
which now set the wage at $3.80 . 21st century ( edit ) Legislation ( edit ) Play media An opinion video from the Economic Policy Institute about working for
the tipped minimum wage of $2.13 per hour . In 2006 , voters in six states ( Arizona , Colorado , Missouri , Montana , Nevada , and Ohio ) approved statewide
increases in the state minimum wage . The amounts of these increases ranged from $1 to $1.70 per hour and all increases were designed to annually index to
inflation .
</item 1>

<item 2>
Minimum wage in the United States
Federal ( edit ) Type Min . wage ( $ / h ) Notes Tipped $2.13 The Fair Labor Standards Act of 1938 has been requiring a minimum wage of $2.13 for tipped workers
with the expectation that wages plus tips total no less than $7.25 per hour since September 1 , 1991 . The employer must pay the difference if total income does
not add up to $7.25 per hour . Non-tipped $7.25 Per the Fair Minimum Wage Act of 2007 ( FMWA ) since July 24 , 2009 . Youth $4.25 The Fair Labor Standards Act
has since September 1 , 1996 allowed for persons under the age of 20 to be paid $4.25 for the first 90 calendar days of their employment . State ( edit ) The
average US minimum wage per capita ( 2017 ) is $8.49 based on the population size of each state and generally represents the average minimum wage experienced by
a person working in one of the fifty US states . Cities , counties , districts , and territories are not included in the calculation .
</item 2>

<item 3>
Minimum wage in the United States
Minimum wage in the United States - wikipedia Minimum wage in the United States Jump to : navigation , search Minimum wage by U.S. state and territory , as of
January 1 , 2018 . Versus the federal rate : Higher Same Lower No state minimum wage law The minimum wage in the United States is set by US labor law and a
range of state and local laws . Employers generally have to pay workers the highest minimum wage prescribed by federal , state , and local law . Since July 24 ,
2009 , the federal government has mandated a nationwide minimum wage of $7.25 per hour . As of January 2018 , there were 29 states with a minimum wage higher
than the federal minimum . From 2017 to 2018 , eight states increased their minimum wage levels through automatic adjustments , while increases in eleven other
states occurred through referendum or legislative action . Washington has the highest state minimum wage at $11.50 per hour . Using 2017 inflation - adjusted
dollars , the federal minimum wage peaked at $11.53 per hour in 1968 .
</item 3>

<item 4>
Minimum wage in the United States
From January 1981 to April 1990 , the minimum wage was frozen at $3.35 per hour , then a record - setting minimum wage freeze . From September 1 , 1997 through
July 23 , 2007 , the federal minimum wage remained constant at $5.15 per hour , breaking the old record . From the United States Department of Labor .
Employment Standards Administration . Wage and Hour Division , the source page has a clickable US map with current and projected state - by - state minimum wage
rates for each state . Some government entities , such as counties and cities , observe minimum wages that are higher than the state as a whole . One notable
example of this is Santa Fe , New Mexico , whose $9.50 per hour minimum wage was the highest in the nation , until San Francisco increased its minimum wage to
$9.79 in 2009 . Another device to increase wages , living wage ordinances , generally apply only to businesses that are under contract to the local government
itself . Since 1984 , the purchasing power of the federal minimum wage has decreased .
</item 4>

<item 5>
Minimum wage in the United States
May 2006 . Retrieved 2013 - 10 - 06 . ^ Jump up to : `` National Occupational Employment and Wage Estimates United States '' . US Bureau of Labor Statistics .
May 2008 . Retrieved 2013 - 10 - 06 . Jump up ^ `` National Occupational Employment and Wage Estimates United States '' . US Bureau of Labor Statistics . May
2009 . Retrieved 2013 - 10 - 06 . External links ( edit ) Minimum wage in the United States at Curlie ( based on DMOZ ) Federal Minimum Wage . United States
Department of Labor Wage and Hour Division . Minimum Wages for Tipped Employees . United States Department of Labor Wage and Hour Division . History of Federal
Minimum Wage United States Department of Labor Wage and Hour Division . U.S. Minimum Wage History . Oregon State University -- Wealth and Poverty ( Anth 484 ) .
Last updated December 26 , 2012 . Maps and charts of minimum wage vs. housing costs - National Low Income Housing Coalition ( advocacy group ) United States
state - related lists List of states and territories of the United States Demographics
</item 5>

<item 6>
Minimum wage in the United States
The Annals of the American Academy of Political and Social Science . 48 : 27 . Jump up ^ Broda , Rudolf ; United States . Bureau of Labor Statistics ( December
1928 ) . Minimum Wage Legislation in Various Countries : Bulletin of the United States Bureau of Labor Statistics , No. 467 ( PDF ) . Bulletin of the United
States Bureau of Labor Statistics . G.P.O. p. 11 . Jump up ^ Jerold L. , , Waltman ( 2000 ) . The politics of the minimum wage . Urbana : University of Illinois
Press . p. 11 . ISBN 9780252025457 . OCLC 42072067 . Jump up ^ Vivien. , Hart , ( 2001 ) . Bound by Our Constitution : Women , Workers , and the Minimum Wage .
Princeton : Princeton University Press . p. 63 . ISBN 9781400821563 . OCLC 700688619 . Jump up ^ Vivien. , Hart , ( 2001 ) .
</item 6>

<item 7>
Minimum wage in the United States
Bureau of Labor Statistics '' . Bls.gov . Retrieved 18 October 2017 . Jump up ^ Tritch , Teresa ( March 7 , 2014 ) . `` F.D.R. Makes the Case for the Minimum
Wage '' . New York Times . Retrieved March 7 , 2014 . Jump up ^ Broda , Rudolf ; United States . Bureau of Labor Statistics ( December 1928 ) . Minimum Wage
Legislation in Various Countries : Bulletin of the United States Bureau of Labor Statistics , No. 467 ( PDF ) . Bulletin of the United States Bureau of Labor
Statistics . G.P.O. p. 8 . Jump up ^ Willis J. , Nordlund ( 1997 ) . The quest for a living wage : the history of the federal minimum wage program . Westport ,
Conn. : Greenwood Press . p. 2 . ISBN 9780313264122 . OCLC 33983425 . Jump up ^ Hammond , Matthew B. ( 1913 ) . `` The Minimum Wage in Great Britain and
Australia '' .
</item 7>

<item 8>
Minimum wage in the United States
However , the Court did not extend this principle to minimum wage laws , considering the latter as interfering with the ability of employers to freely negotiate
wage contracts with employees . In 1933 , the Roosevelt administration made the first attempt at establishing a national minimum wage , when a $0.25 per hour
standard was set as part of the National Industrial Recovery Act . However , in the 1935 court case Schechter Poultry Corp. v. United States ( 295 U.S. 495 ) ,
the US Supreme Court declared the act unconstitutional , and the minimum wage was abolished . In 1938 , the minimum wage was re-established pursuant to the Fair
Labor Standards Act , once again at $0.25 per hour ( $4.78 in 2017 dollars ) . In 1941 , the Supreme Court upheld the Fair Labor Standards Act in United States
v. Darby Lumber Co. , holding that Congress had the power under the Commerce Clause to regulate employment conditions .
</item 8>

<item 9>
Minimum wage in the United States
2 Living wage protests 1.3. 3 Local ordinances 1.3. 4 Union exemptions 2 Trends in purchasing power 3 Economic effects 3.1 Employment and job creation 3.2
Prices 3.3 Effects on crime 3.4 Economic growth 3.5 Income inequality 3.6 Poverty 3.7 Federal budget deficit 4 Commentary 4.1 Economists 4.2 Major political
parties 5 Polls 6 List by jurisdiction 6.1 Federal 6.2 State 6.3 Territory 6.4 Federal District 7 Low - paying occupations , 2006 and 2009 8 See also 9
References 10 External links History ( edit ) 19th century ( edit ) No business which depends for existence on paying less than living wages to its workers has
any right to continue in this country . President Franklin Delano Roosevelt , 1933 Minimum wage legislation emerged at the end of the nineteenth century from
the desire to end sweated labor which had developed in the wake of industrialization .
</item 9>

<item 10>
Minimum wage in the United States
20th century ( edit ) In 1910 , in conjunction with advocacy work led by Florence Kelley of the National Consumer League , the Women 's Trade Union League (
WTLU ) of Massachusetts under the leadership of Elizabeth Evans took up the cause of minimum wage legislation in Massachusetts . Over the next two years , a
coalition of social reform groups and labor advocates in Boston pushed for minimum wage legislation in the state . In 1912 , Massachusetts passed the first
American minimum wage legislation , which established a state commission for recommending non-compulsory minimum wages for women and children . Within eight
years , at least thirteen U.S. states and the District of Columbia would pass minimum wage laws , with pressure being placed on state legislatures by the
National Consumers League in a coalition with other women 's voluntary associations and organized labor . The United States Supreme Court of the Lochner era
consistently invalidated compulsory minimum wage laws . Advocates for these minimum wage laws hoped that they would be upheld under the precedent of Muller v.
Oregon , which upheld maximum working hours laws for women on the grounds that women required special protection that men did not .
</item 10>
=== END ITEM id=318df246095ee282 ===

=== ITEM id=ca61de502d81fe8c ===
Gold sentences and facts:
Sentence 0: Country Adult prevalence of HIV / AIDS Number of people with HIV / AIDS Annual deaths from HIV / AIDS Year of estimate Swaziland 27.20 % 220,000 3,900 2016 Lesotho 25.00 % 330,000 9,900 2016 Botswana 21.90 % 360,000 3,900 2016 South Africa 18.90 % 7,100,000 110,000 2016 Namibia 13.80 % 230,000 4,300 2016 Zimbabwe 13.50 % 1,300,000 30,000 2016 Zambia 12.40 % 1,200,000 21,000 2016 Mozambique 12.30 % 1,800,000 62,000 2016 Malawi 9.20 % 1,000,000 24,000 2016 Uganda 6.50 % 1,400,000 28,000 2016 Equatorial Guinea 6.20 % 35,000 - 2016 Kenya 5.40 % 1,600,000 36,000 2016 Tanzania 4.70 % 1,400,000 33,000 2016 Central African Republic 4.00 % 130,000 7,300 2016 Cameroon 3.80 % 560,000 29,000 2016 Gabon 3.60 % 48,000 1,500 2016 Bahamas , The 3.30 % 8,200 - 2016 Rwanda 3.10 % 220,000 3,300 2016 Congo , Republic of the 3.10 % 91,000 3,800 2016 Guinea - Bissau 3.10 % 36,000 2,000 2016 Nigeria 2.90 % 3,200,000 160,000 2016 Cote d'Ivoire 2.70 % 460,000 25,000 2016 South Sudan 2.70 % 200,000 13,000 2016 Haiti 2.10 % 150,000 4,600 2016 Togo 2.10 % 100,000 5,100 2016 Angola 1.90 % 280,000 11,000 2016 Belize 1.80 % 4,300 - 2016 Sierra Leone 1.70 % 67,000 2,800 2016 Gambia , The 1.70 % 20,000 1,100 2016 Jamaica 1.70 % 30,000 1,300 2016 Liberia 1.60 % 43,000 2,800 2016 Guyana 1.60 % 8,500 - 2016 Ghana 1.60 % 290,000 15,000 2016 Guinea 1.50 % 120,000 5,800 2016 Suriname 1.40 % 4,900 - 2016 Barbados 1.30 % 2,600 - 2016 Djibouti 1.30 % 8,600 - 2016 Chad 1.30 % 110,000 2,800 2016 Trinidad and Tobago 1.20 % 11,000 - 2016 Thailand 1.10 % 450,000 16,000 2016 Ethiopia 1.10 % 710,000 20,000 2016 Burundi 1.10 % 84,000 2,900 2016 Benin 1.00 % 67,000 2,400 2016 Mali 1.00 % 110,000 6,100 2016 Dominican Republic 1.00 % 67,000 2,200 2016 Ukraine 0.90 % 240,000 8,500 2016 Papua New Guinea 0.90 % 46,000 1,100 2016 Cabo Verde 0.80 % 2,800 - 2016 Panama 0.80 % 21,000 - 2016 Burma 0.80 % 230,000 7,800 2016 Burkina Faso 0.80 % 95,000 3,100 2016 Latvia 0.70 % 6,600 - 2016 Congo , Democratic Republic of 0.70 % 370,000 19,000 2016 Cambodia 0.60 % 71,000 1,800 2016 Venezuela 0.60 % 120,000 2,500 2016 Moldova 0.60 % 15,000 - 2016 Eritrea 0.60 % 15,000 - 2016 El Salvador 0.60 % 24,000 - 2016 Uruguay 0.60 % 12,000 - 2016 Brazil 0.60 % 830,000 14,000 2016 Georgia 0.50 % 12,000 - 2016 Mauritania 0.50 % 11,000 - 2016 Chile 0.50 % 61,000 - 2016 Guatemala 0.50 % 46,000 1,600 2016 Paraguay 0.50 % 19,000 - 2016 Niger 0.40 % 48,000 3,400 2016 Malaysia 0.40 % 97,000 7,000 2016 Indonesia 0.40 % 620,000 38,000 2016 Honduras 0.40 % 21,000 - 2016 Canada 0.40 % 180,000 - 2016 Cuba 0.40 % 25,000 - 2016 Costa Rica 0.40 % 13,000 - 2016 Colombia 0.40 % 120,000 2,800 2016 Vietnam 0.40 % 250,000 8,000 2016 Argentina 0.40 % 120,000 2,400 2016 Belarus 0.40 % 19,000 - 2016 USA 0.30 % 1,100,000 - 2015 Somalia 2016 Senegal 0.40 % 41,000 1,900 2016 India 0.30 % 2,100,000 62,000 2016 Ecuador 0.30 % 33,000 - 2016 Bolivia 0.30 % 19,000 - 2016 Tajikistan 0.30 % 14,000 - 2016 Peru 0.30 % 70,000 2,200 2016 Italy 0.30 % 130,000 - 2016 Laos 0.30 % 11,000 - 2016 Mexico 0.30 % 220,000 4,200 2016 Armenia 0.20 % 3,300 - 2016 Ireland 0.20 % 6,200 - 2016 Kazakhstan 0.20 % 26,000 - 2016 Kyrgyzstan 0.20 % 8,500 - 2016 Lithuania 0.20 % 2,900 - 2016 Madagascar 0.20 % 31,000 1,600 2016 Nepal 0.20 % 32,000 1,700 2016 Netherlands 0.20 % 23,000 - 2016 Nicaragua 0.20 % 8,900 - 2016 Sudan 0.20 % 56,000 3,000 2016 Sweden 0.20 % 11,000 - 2016 Albania 0.10 % 1,700 - 2016 Australia 0.10 % 25,000 - 2016 Fiji 0.10 % - - 2016 Taiwan 0.10 % 31,620 - 2016 Azerbaijan 0.10 % 9,200 - 2016 Iran 0.10 % 66,000 4,000 2016 Romania 0.10 % 16,000 - 2016 Montenegro 0.10 % - - 2016 Morocco 0.10 % 22,000 - 2016 Pakistan 0.10 % 130,000 5,500 2016 Philippines 0.10 % 56,000 - 2016 Malta 0.10 % - - 2016 Algeria - 13,000 - 2016 Bangladesh - 12,000 1,000 2016 Egypt - 11,000 - 2016 Yemen - 9,900 - 2016 Saudi Arabia - 8,200 - 2016 Afghanistan - 7,500 - 2016 Sri Lanka - 4,000 - 2016 Bulgaria - 3,500 - 2016 Czech Republic - 3,400 - 2016 Tunisia - 2,900 - 2016 Serbia - 2,700 - 2016 Lebanon - 2,200 - 2016 Croatia - 1,500 - 2016 Bosnia 0.009 % 350 - 2017
  Fact 0.0: The table gives, for each country, the adult prevalence of HIV/AIDS, the number of people with HIV/AIDS, the annual deaths from HIV/AIDS, and the year of estimate.
  Fact 0.1: The year of estimate for Somalia is 2016.
  Fact 0.2: In 2016, the adult HIV/AIDS prevalence in Swaziland was 27.20%.
  Fact 0.3: In 2016, Swaziland had 220,000 people with HIV/AIDS.
  Fact 0.4: In 2016, there were 3,900 annual deaths from HIV/AIDS in Swaziland.
  Fact 0.5: In 2016, the adult HIV/AIDS prevalence in Lesotho was 25.00%.
  Fact 0.6: In 2016, Lesotho had 330,000 people with HIV/AIDS.
  Fact 0.7: In 2016, there were 9,900 annual deaths from HIV/AIDS in Lesotho.
  Fact 0.8: In 2016, the adult HIV/AIDS prevalence in Botswana was 21.90%.
  Fact 0.9: In 2016, Botswana had 360,000 people with HIV/AIDS.
  Fact 0.10: In 2016, there were 3,900 annual deaths from HIV/AIDS in Botswana.
  Fact 0.11: In 2016, the adult HIV/AIDS prevalence in South Africa was 18.90%.
  Fact 0.12: In 2016, South Africa had 7,100,000 people with HIV/AIDS.
  Fact 0.13: In 2016, there were 110,000 annual deaths from HIV/AIDS in South Africa.
  Fact 0.14: In 2016, the adult HIV/AIDS prevalence in Namibia was 13.80%.
  Fact 0.15: In 2016, Namibia had 230,000 people with HIV/AIDS.
  Fact 0.16: In 2016, there were 4,300 annual deaths from HIV/AIDS in Namibia.
  Fact 0.17: In 2016, the adult HIV/AIDS prevalence in Zimbabwe was 13.50%.
  Fact 0.18: In 2016, Zimbabwe had 1,300,000 people with HIV/AIDS.
  Fact 0.19: In 2016, there were 30,000 annual deaths from HIV/AIDS in Zimbabwe.
  Fact 0.20: In 2016, the adult HIV/AIDS prevalence in Zambia was 12.40%.
  Fact 0.21: In 2016, Zambia had 1,200,000 people with HIV/AIDS.
  Fact 0.22: In 2016, there were 21,000 annual deaths from HIV/AIDS in Zambia.
  Fact 0.23: In 2016, the adult HIV/AIDS prevalence in Mozambique was 12.30%.
  Fact 0.24: In 2016, Mozambique had 1,800,000 people with HIV/AIDS.
  Fact 0.25: In 2016, there were 62,000 annual deaths from HIV/AIDS in Mozambique.
  Fact 0.26: In 2016, the adult HIV/AIDS prevalence in Malawi was 9.20%.
  Fact 0.27: In 2016, Malawi had 1,000,000 people with HIV/AIDS.
  Fact 0.28: In 2016, there were 24,000 annual deaths from HIV/AIDS in Malawi.
  Fact 0.29: In 2016, the adult HIV/AIDS prevalence in Uganda was 6.50%.
  Fact 0.30: In 2016, Uganda had 1,400,000 people with HIV/AIDS.
  Fact 0.31: In 2016, there were 28,000 annual deaths from HIV/AIDS in Uganda.
  Fact 0.32: In 2016, the adult HIV/AIDS prevalence in Equatorial Guinea was 6.20%.
  Fact 0.33: In 2016, Equatorial Guinea had 35,000 people with HIV/AIDS.
  Fact 0.34: In 2016, the adult HIV/AIDS prevalence in Kenya was 5.40%.
  Fact 0.35: In 2016, Kenya had 1,600,000 people with HIV/AIDS.
  Fact 0.36: In 2016, there were 36,000 annual deaths from HIV/AIDS in Kenya.
  Fact 0.37: In 2016, the adult HIV/AIDS prevalence in Tanzania was 4.70%.
  Fact 0.38: In 2016, Tanzania had 1,400,000 people with HIV/AIDS.
  Fact 0.39: In 2016, there were 33,000 annual deaths from HIV/AIDS in Tanzania.
  Fact 0.40: In 2016, the adult HIV/AIDS prevalence in Central African Republic was 4.00%.
  Fact 0.41: In 2016, Central African Republic had 130,000 people with HIV/AIDS.
  Fact 0.42: In 2016, there were 7,300 annual deaths from HIV/AIDS in Central African Republic.
  Fact 0.43: In 2016, the adult HIV/AIDS prevalence in Cameroon was 3.80%.
  Fact 0.44: In 2016, Cameroon had 560,000 people with HIV/AIDS.
  Fact 0.45: In 2016, there were 29,000 annual deaths from HIV/AIDS in Cameroon.
  Fact 0.46: In 2016, the adult HIV/AIDS prevalence in Gabon was 3.60%.
  Fact 0.47: In 2016, Gabon had 48,000 people with HIV/AIDS.
  Fact 0.48: In 2016, there were 1,500 annual deaths from HIV/AIDS in Gabon.
  Fact 0.49: In 2016, the adult HIV/AIDS prevalence in Bahamas, The was 3.30%.
  Fact 0.50: In 2016, Bahamas, The had 8,200 people with HIV/AIDS.
  Fact 0.51: In 2016, the adult HIV/AIDS prevalence in Rwanda was 3.10%.
  Fact 0.52: In 2016, Rwanda had 220,000 people with HIV/AIDS.
  Fact 0.53: In 2016, there were 3,300 annual deaths from HIV/AIDS in Rwanda.
  Fact 0.54: In 2016, the adult HIV/AIDS prevalence in Congo, Republic of the was 3.10%.
  Fact 0.55: In 2016, Congo, Republic of the had 91,000 people with HIV/AIDS.
  Fact 0.56: In 2016, there were 3,800 annual deaths from HIV/AIDS in Congo, Republic of the.
  Fact 0.57: In 2016, the adult HIV/AIDS prevalence in Guinea-Bissau was 3.10%.
  Fact 0.58: In 2016, Guinea-Bissau had 36,000 people with HIV/AIDS.
  Fact 0.59: In 2016, there were 2,000 annual deaths from HIV/AIDS in Guinea-Bissau.
  Fact 0.60: In 2016, the adult HIV/AIDS prevalence in Nigeria was 2.90%.
  Fact 0.61: In 2016, Nigeria had 3,200,000 people with HIV/AIDS.
  Fact 0.62: In 2016, there were 160,000 annual deaths from HIV/AIDS in Nigeria.
  Fact 0.63: In 2016, the adult HIV/AIDS prevalence in Cote d'Ivoire was 2.70%.
  Fact 0.64: In 2016, Cote d'Ivoire had 460,000 people with HIV/AIDS.
  Fact 0.65: In 2016, there were 25,000 annual deaths from HIV/AIDS in Cote d'Ivoire.
  Fact 0.66: In 2016, the adult HIV/AIDS prevalence in South Sudan was 2.70%.
  Fact 0.67: In 2016, South Sudan had 200,000 people with HIV/AIDS.
  Fact 0.68: In 2016, there were 13,000 annual deaths from HIV/AIDS in South Sudan.
  Fact 0.69: In 2016, the adult HIV/AIDS prevalence in Haiti was 2.10%.
  Fact 0.70: In 2016, Haiti had 150,000 people with HIV/AIDS.
  Fact 0.71: In 2016, there were 4,600 annual deaths from HIV/AIDS in Haiti.
  Fact 0.72: In 2016, the adult HIV/AIDS prevalence in Togo was 2.10%.
  Fact 0.73: In 2016, Togo had 100,000 people with HIV/AIDS.
  Fact 0.74: In 2016, there were 5,100 annual deaths from HIV/AIDS in Togo.
  Fact 0.75: In 2016, the adult HIV/AIDS prevalence in Angola was 1.90%.
  Fact 0.76: In 2016, Angola had 280,000 people with HIV/AIDS.
  Fact 0.77: In 2016, there were 11,000 annual deaths from HIV/AIDS in Angola.
  Fact 0.78: In 2016, the adult HIV/AIDS prevalence in Belize was 1.80%.
  Fact 0.79: In 2016, Belize had 4,300 people with HIV/AIDS.
  Fact 0.80: In 2016, the adult HIV/AIDS prevalence in Sierra Leone was 1.70%.
  Fact 0.81: In 2016, Sierra Leone had 67,000 people with HIV/AIDS.
  Fact 0.82: In 2016, there were 2,800 annual deaths from HIV/AIDS in Sierra Leone.
  Fact 0.83: In 2016, the adult HIV/AIDS prevalence in Gambia, The was 1.70%.
  Fact 0.84: In 2016, Gambia, The had 20,000 people with HIV/AIDS.
  Fact 0.85: In 2016, there were 1,100 annual deaths from HIV/AIDS in Gambia, The.
  Fact 0.86: In 2016, the adult HIV/AIDS prevalence in Jamaica was 1.70%.
  Fact 0.87: In 2016, Jamaica had 30,000 people with HIV/AIDS.
  Fact 0.88: In 2016, there were 1,300 annual deaths from HIV/AIDS in Jamaica.
  Fact 0.89: In 2016, the adult HIV/AIDS prevalence in Liberia was 1.60%.
  Fact 0.90: In 2016, Liberia had 43,000 people with HIV/AIDS.
  Fact 0.91: In 2016, there were 2,800 annual deaths from HIV/AIDS in Liberia.
  Fact 0.92: In 2016, the adult HIV/AIDS prevalence in Guyana was 1.60%.
  Fact 0.93: In 2016, Guyana had 8,500 people with HIV/AIDS.
  Fact 0.94: In 2016, the adult HIV/AIDS prevalence in Ghana was 1.60%.
  Fact 0.95: In 2016, Ghana had 290,000 people with HIV/AIDS.
  Fact 0.96: In 2016, there were 15,000 annual deaths from HIV/AIDS in Ghana.
  Fact 0.97: In 2016, the adult HIV/AIDS prevalence in Guinea was 1.50%.
  Fact 0.98: In 2016, Guinea had 120,000 people with HIV/AIDS.
  Fact 0.99: In 2016, there were 5,800 annual deaths from HIV/AIDS in Guinea.
  Fact 0.100: In 2016, the adult HIV/AIDS prevalence in Suriname was 1.40%.
  Fact 0.101: In 2016, Suriname had 4,900 people with HIV/AIDS.
  Fact 0.102: In 2016, the adult HIV/AIDS prevalence in Barbados was 1.30%.
  Fact 0.103: In 2016, Barbados had 2,600 people with HIV/AIDS.
  Fact 0.104: In 2016, the adult HIV/AIDS prevalence in Djibouti was 1.30%.
  Fact 0.105: In 2016, Djibouti had 8,600 people with HIV/AIDS.
  Fact 0.106: In 2016, the adult HIV/AIDS prevalence in Chad was 1.30%.
  Fact 0.107: In 2016, Chad had 110,000 people with HIV/AIDS.
  Fact 0.108: In 2016, there were 2,800 annual deaths from HIV/AIDS in Chad.
  Fact 0.109: In 2016, the adult HIV/AIDS prevalence in Trinidad and Tobago was 1.20%.
  Fact 0.110: In 2016, Trinidad and Tobago had 11,000 people with HIV/AIDS.
  Fact 0.111: In 2016, the adult HIV/AIDS prevalence in Thailand was 1.10%.
  Fact 0.112: In 2016, Thailand had 450,000 people with HIV/AIDS.
  Fact 0.113: In 2016, there were 16,000 annual deaths from HIV/AIDS in Thailand.
  Fact 0.114: In 2016, the adult HIV/AIDS prevalence in Ethiopia was 1.10%.
  Fact 0.115: In 2016, Ethiopia had 710,000 people with HIV/AIDS.
  Fact 0.116: In 2016, there were 20,000 annual deaths from HIV/AIDS in Ethiopia.
  Fact 0.117: In 2016, the adult HIV/AIDS prevalence in Burundi was 1.10%.
  Fact 0.118: In 2016, Burundi had 84,000 people with HIV/AIDS.
  Fact 0.119: In 2016, there were 2,900 annual deaths from HIV/AIDS in Burundi.
  Fact 0.120: In 2016, the adult HIV/AIDS prevalence in Benin was 1.00%.
  Fact 0.121: In 2016, Benin had 67,000 people with HIV/AIDS.
  Fact 0.122: In 2016, there were 2,400 annual deaths from HIV/AIDS in Benin.
  Fact 0.123: In 2016, the adult HIV/AIDS prevalence in Mali was 1.00%.
  Fact 0.124: In 2016, Mali had 110,000 people with HIV/AIDS.
  Fact 0.125: In 2016, there were 6,100 annual deaths from HIV/AIDS in Mali.
  Fact 0.126: In 2016, the adult HIV/AIDS prevalence in Dominican Republic was 1.00%.
  Fact 0.127: In 2016, Dominican Republic had 67,000 people with HIV/AIDS.
  Fact 0.128: In 2016, there were 2,200 annual deaths from HIV/AIDS in Dominican Republic.
  Fact 0.129: In 2016, the adult HIV/AIDS prevalence in Ukraine was 0.90%.
  Fact 0.130: In 2016, Ukraine had 240,000 people with HIV/AIDS.
  Fact 0.131: In 2016, there were 8,500 annual deaths from HIV/AIDS in Ukraine.
  Fact 0.132: In 2016, the adult HIV/AIDS prevalence in Papua New Guinea was 0.90%.
  Fact 0.133: In 2016, Papua New Guinea had 46,000 people with HIV/AIDS.
  Fact 0.134: In 2016, there were 1,100 annual deaths from HIV/AIDS in Papua New Guinea.
  Fact 0.135: In 2016, the adult HIV/AIDS prevalence in Cabo Verde was 0.80%.
  Fact 0.136: In 2016, Cabo Verde had 2,800 people with HIV/AIDS.
  Fact 0.137: In 2016, the adult HIV/AIDS prevalence in Panama was 0.80%.
  Fact 0.138: In 2016, Panama had 21,000 people with HIV/AIDS.
  Fact 0.139: In 2016, the adult HIV/AIDS prevalence in Burma was 0.80%.
  Fact 0.140: In 2016, Burma had 230,000 people with HIV/AIDS.
  Fact 0.141: In 2016, there were 7,800 annual deaths from HIV/AIDS in Burma.
  Fact 0.142: In 2016, the adult HIV/AIDS prevalence in Burkina Faso was 0.80%.
  Fact 0.143: In 2016, Burkina Faso had 95,000 people with HIV/AIDS.
  Fact 0.144: In 2016, there were 3,100 annual deaths from HIV/AIDS in Burkina Faso.
  Fact 0.145: In 2016, the adult HIV/AIDS prevalence in Latvia was 0.70%.
  Fact 0.146: In 2016, Latvia had 6,600 people with HIV/AIDS.
  Fact 0.147: In 2016, the adult HIV/AIDS prevalence in Congo, Democratic Republic of was 0.70%.
  Fact 0.148: In 2016, Congo, Democratic Republic of had 370,000 people with HIV/AIDS.
  Fact 0.149: In 2016, there were 19,000 annual deaths from HIV/AIDS in Congo, Democratic Republic of.
  Fact 0.150: In 2016, the adult HIV/AIDS prevalence in Cambodia was 0.60%.
  Fact 0.151: In 2016, Cambodia had 71,000 people with HIV/AIDS.
  Fact 0.152: In 2016, there were 1,800 annual deaths from HIV/AIDS in Cambodia.
  Fact 0.153: In 2016, the adult HIV/AIDS prevalence in Venezuela was 0.60%.
  Fact 0.154: In 2016, Venezuela had 120,000 people with HIV/AIDS.
  Fact 0.155: In 2016, there were 2,500 annual deaths from HIV/AIDS in Venezuela.
  Fact 0.156: In 2016, the adult HIV/AIDS prevalence in Moldova was 0.60%.
  Fact 0.157: In 2016, Moldova had 15,000 people with HIV/AIDS.
  Fact 0.158: In 2016, the adult HIV/AIDS prevalence in Eritrea was 0.60%.
  Fact 0.159: In 2016, Eritrea had 15,000 people with HIV/AIDS.
  Fact 0.160: In 2016, the adult HIV/AIDS prevalence in El Salvador was 0.60%.
  Fact 0.161: In 2016, El Salvador had 24,000 people with HIV/AIDS.
  Fact 0.162: In 2016, the adult HIV/AIDS prevalence in Uruguay was 0.60%.
  Fact 0.163: In 2016, Uruguay had 12,000 people with HIV/AIDS.
  Fact 0.164: In 2016, the adult HIV/AIDS prevalence in Brazil was 0.60%.
  Fact 0.165: In 2016, Brazil had 830,000 people with HIV/AIDS.
  Fact 0.166: In 2016, there were 14,000 annual deaths from HIV/AIDS in Brazil.
  Fact 0.167: In 2016, the adult HIV/AIDS prevalence in Georgia was 0.50%.
  Fact 0.168: In 2016, Georgia had 12,000 people with HIV/AIDS.
  Fact 0.169: In 2016, the adult HIV/AIDS prevalence in Mauritania was 0.50%.
  Fact 0.170: In 2016, Mauritania had 11,000 people with HIV/AIDS.
  Fact 0.171: In 2016, the adult HIV/AIDS prevalence in Chile was 0.50%.
  Fact 0.172: In 2016, Chile had 61,000 people with HIV/AIDS.
  Fact 0.173: In 2016, the adult HIV/AIDS prevalence in Guatemala was 0.50%.
  Fact 0.174: In 2016, Guatemala had 46,000 people with HIV/AIDS.
  Fact 0.175: In 2016, there were 1,600 annual deaths from HIV/AIDS in Guatemala.
  Fact 0.176: In 2016, the adult HIV/AIDS prevalence in Paraguay was 0.50%.
  Fact 0.177: In 2016, Paraguay had 19,000 people with HIV/AIDS.
  Fact 0.178: In 2016, the adult HIV/AIDS prevalence in Niger was 0.40%.
  Fact 0.179: In 2016, Niger had 48,000 people with HIV/AIDS.
  Fact 0.180: In 2016, there were 3,400 annual deaths from HIV/AIDS in Niger.
  Fact 0.181: In 2016, the adult HIV/AIDS prevalence in Malaysia was 0.40%.
  Fact 0.182: In 2016, Malaysia had 97,000 people with HIV/AIDS.
  Fact 0.183: In 2016, there were 7,000 annual deaths from HIV/AIDS in Malaysia.
  Fact 0.184: In 2016, the adult HIV/AIDS prevalence in Indonesia was 0.40%.
  Fact 0.185: In 2016, Indonesia had 620,000 people with HIV/AIDS.
  Fact 0.186: In 2016, there were 38,000 annual deaths from HIV/AIDS in Indonesia.
  Fact 0.187: In 2016, the adult HIV/AIDS prevalence in Honduras was 0.40%.
  Fact 0.188: In 2016, Honduras had 21,000 people with HIV/AIDS.
  Fact 0.189: In 2016, the adult HIV/AIDS prevalence in Canada was 0.40%.
  Fact 0.190: In 2016, Canada had 180,000 people with HIV/AIDS.
  Fact 0.191: In 2016, the adult HIV/AIDS prevalence in Cuba was 0.40%.
  Fact 0.192: In 2016, Cuba had 25,000 people with HIV/AIDS.
  Fact 0.193: In 2016, the adult HIV/AIDS prevalence in Costa Rica was 0.40%.
  Fact 0.194: In 2016, Costa Rica had 13,000 people with HIV/AIDS.
  Fact 0.195: In 2016, the adult HIV/AIDS prevalence in Colombia was 0.40%.
  Fact 0.196: In 2016, Colombia had 120,000 people with HIV/AIDS.
  Fact 0.197: In 2016, there were 2,800 annual deaths from HIV/AIDS in Colombia.
  Fact 0.198: In 2016, the adult HIV/AIDS prevalence in Vietnam was 0.40%.
  Fact 0.199: In 2016, Vietnam had 250,000 people with HIV/AIDS.
  Fact 0.200: In 2016, there were 8,000 annual deaths from HIV/AIDS in Vietnam.
  Fact 0.201: In 2016, the adult HIV/AIDS prevalence in Argentina was 0.40%.
  Fact 0.202: In 2016, Argentina had 120,000 people with HIV/AIDS.
  Fact 0.203: In 2016, there were 2,400 annual deaths from HIV/AIDS in Argentina.
  Fact 0.204: In 2016, the adult HIV/AIDS prevalence in Belarus was 0.40%.
  Fact 0.205: In 2016, Belarus had 19,000 people with HIV/AIDS.
  Fact 0.206: In 2015, the adult HIV/AIDS prevalence in USA was 0.30%.
  Fact 0.207: In 2015, USA had 1,100,000 people with HIV/AIDS.
  Fact 0.208: In 2016, the adult HIV/AIDS prevalence in Senegal was 0.40%.
  Fact 0.209: In 2016, Senegal had 41,000 people with HIV/AIDS.
  Fact 0.210: In 2016, there were 1,900 annual deaths from HIV/AIDS in Senegal.
  Fact 0.211: In 2016, the adult HIV/AIDS prevalence in India was 0.30%.
  Fact 0.212: In 2016, India had 2,100,000 people with HIV/AIDS.
  Fact 0.213: In 2016, there were 62,000 annual deaths from HIV/AIDS in India.
  Fact 0.214: In 2016, the adult HIV/AIDS prevalence in Ecuador was 0.30%.
  Fact 0.215: In 2016, Ecuador had 33,000 people with HIV/AIDS.
  Fact 0.216: In 2016, the adult HIV/AIDS prevalence in Bolivia was 0.30%.
  Fact 0.217: In 2016, Bolivia had 19,000 people with HIV/AIDS.
  Fact 0.218: In 2016, the adult HIV/AIDS prevalence in Tajikistan was 0.30%.
  Fact 0.219: In 2016, Tajikistan had 14,000 people with HIV/AIDS.
  Fact 0.220: In 2016, the adult HIV/AIDS prevalence in Peru was 0.30%.
  Fact 0.221: In 2016, Peru had 70,000 people with HIV/AIDS.
  Fact 0.222: In 2016, there were 2,200 annual deaths from HIV/AIDS in Peru.
  Fact 0.223: In 2016, the adult HIV/AIDS prevalence in Italy was 0.30%.
  Fact 0.224: In 2016, Italy had 130,000 people with HIV/AIDS.
  Fact 0.225: In 2016, the adult HIV/AIDS prevalence in Laos was 0.30%.
  Fact 0.226: In 2016, Laos had 11,000 people with HIV/AIDS.
  Fact 0.227: In 2016, the adult HIV/AIDS prevalence in Mexico was 0.30%.
  Fact 0.228: In 2016, Mexico had 220,000 people with HIV/AIDS.
  Fact 0.229: In 2016, there were 4,200 annual deaths from HIV/AIDS in Mexico.
  Fact 0.230: In 2016, the adult HIV/AIDS prevalence in Armenia was 0.20%.
  Fact 0.231: In 2016, Armenia had 3,300 people with HIV/AIDS.
  Fact 0.232: In 2016, the adult HIV/AIDS prevalence in Ireland was 0.20%.
  Fact 0.233: In 2016, Ireland had 6,200 people with HIV/AIDS.
  Fact 0.234: In 2016, the adult HIV/AIDS prevalence in Kazakhstan was 0.20%.
  Fact 0.235: In 2016, Kazakhstan had 26,000 people with HIV/AIDS.
  Fact 0.236: In 2016, the adult HIV/AIDS prevalence in Kyrgyzstan was 0.20%.
  Fact 0.237: In 2016, Kyrgyzstan had 8,500 people with HIV/AIDS.
  Fact 0.238: In 2016, the adult HIV/AIDS prevalence in Lithuania was 0.20%.
  Fact 0.239: In 2016, Lithuania had 2,900 people with HIV/AIDS.
  Fact 0.240: In 2016, the adult HIV/AIDS prevalence in Madagascar was 0.20%.
  Fact 0.241: In 2016, Madagascar had 31,000 people with HIV/AIDS.
  Fact 0.242: In 2016, there were 1,600 annual deaths from HIV/AIDS in Madagascar.
  Fact 0.243: In 2016, the adult HIV/AIDS prevalence in Nepal was 0.20%.
  Fact 0.244: In 2016, Nepal had 32,000 people with HIV/AIDS.
  Fact 0.245: In 2016, there were 1,700 annual deaths from HIV/AIDS in Nepal.
  Fact 0.246: In 2016, the adult HIV/AIDS prevalence in Netherlands was 0.20%.
  Fact 0.247: In 2016, Netherlands had 23,000 people with HIV/AIDS.
  Fact 0.248: In 2016, the adult HIV/AIDS prevalence in Nicaragua was 0.20%.
  Fact 0.249: In 2016, Nicaragua had 8,900 people with HIV/AIDS.
  Fact 0.250: In 2016, the adult HIV/AIDS prevalence in Sudan was 0.20%.
  Fact 0.251: In 2016, Sudan had 56,000 people with HIV/AIDS.
  Fact 0.252: In 2016, there were 3,000 annual deaths from HIV/AIDS in Sudan.
  Fact 0.253: In 2016, the adult HIV/AIDS prevalence in Sweden was 0.20%.
  Fact 0.254: In 2016, Sweden had 11,000 people with HIV/AIDS.
  Fact 0.255: In 2016, the adult HIV/AIDS prevalence in Albania was 0.10%.
  Fact 0.256: In 2016, Albania had 1,700 people with HIV/AIDS.
  Fact 0.257: In 2016, the adult HIV/AIDS prevalence in Australia was 0.10%.
  Fact 0.258: In 2016, Australia had 25,000 people with HIV/AIDS.
  Fact 0.259: In 2016, the adult HIV/AIDS prevalence in Fiji was 0.10%.
  Fact 0.260: In 2016, the adult HIV/AIDS prevalence in Taiwan was 0.10%.
  Fact 0.261: In 2016, Taiwan had 31,620 people with HIV/AIDS.
  Fact 0.262: In 2016, the adult HIV/AIDS prevalence in Azerbaijan was 0.10%.
  Fact 0.263: In 2016, Azerbaijan had 9,200 people with HIV/AIDS.
  Fact 0.264: In 2016, the adult HIV/AIDS prevalence in Iran was 0.10%.
  Fact 0.265: In 2016, Iran had 66,000 people with HIV/AIDS.
  Fact 0.266: In 2016, there were 4,000 annual deaths from HIV/AIDS in Iran.
  Fact 0.267: In 2016, the adult HIV/AIDS prevalence in Romania was 0.10%.
  Fact 0.268: In 2016, Romania had 16,000 people with HIV/AIDS.
  Fact 0.269: In 2016, the adult HIV/AIDS prevalence in Montenegro was 0.10%.
  Fact 0.270: In 2016, the adult HIV/AIDS prevalence in Morocco was 0.10%.
  Fact 0.271: In 2016, Morocco had 22,000 people with HIV/AIDS.
  Fact 0.272: In 2016, the adult HIV/AIDS prevalence in Pakistan was 0.10%.
  Fact 0.273: In 2016, Pakistan had 130,000 people with HIV/AIDS.
  Fact 0.274: In 2016, there were 5,500 annual deaths from HIV/AIDS in Pakistan.
  Fact 0.275: In 2016, the adult HIV/AIDS prevalence in Philippines was 0.10%.
  Fact 0.276: In 2016, Philippines had 56,000 people with HIV/AIDS.
  Fact 0.277: In 2016, the adult HIV/AIDS prevalence in Malta was 0.10%.
  Fact 0.278: In 2016, Algeria had 13,000 people with HIV/AIDS.
  Fact 0.279: In 2016, Bangladesh had 12,000 people with HIV/AIDS.
  Fact 0.280: In 2016, there were 1,000 annual deaths from HIV/AIDS in Bangladesh.
  Fact 0.281: In 2016, Egypt had 11,000 people with HIV/AIDS.
  Fact 0.282: In 2016, Yemen had 9,900 people with HIV/AIDS.
  Fact 0.283: In 2016, Saudi Arabia had 8,200 people with HIV/AIDS.
  Fact 0.284: In 2016, Afghanistan had 7,500 people with HIV/AIDS.
  Fact 0.285: In 2016, Sri Lanka had 4,000 people with HIV/AIDS.
  Fact 0.286: In 2016, Bulgaria had 3,500 people with HIV/AIDS.
  Fact 0.287: In 2016, Czech Republic had 3,400 people with HIV/AIDS.
  Fact 0.288: In 2016, Tunisia had 2,900 people with HIV/AIDS.
  Fact 0.289: In 2016, Serbia had 2,700 people with HIV/AIDS.
  Fact 0.290: In 2016, Lebanon had 2,200 people with HIV/AIDS.
  Fact 0.291: In 2016, Croatia had 1,500 people with HIV/AIDS.
  Fact 0.292: In 2017, the adult HIV/AIDS prevalence in Bosnia was 0.009%.
  Fact 0.293: In 2017, Bosnia had 350 people with HIV/AIDS.

Retrieved items:
<item 1>
List of countries by HIV/AIDS adult prevalence rate
Over 10 % of all people living with HIV / AIDS reside within the region . Adult HIV prevalence exceeds 20 % in Swaziland , Botswana , and Lesotho , while an
additional six countries report adult HIV prevalence of at least 10 % . Outside Africa , the highest prevalence rate is found in the Bahamas ( 3.3 % ) . In
absolute numbers , South Africa ( 7.1 million ) , followed by Nigeria ( 3.2 million ) , and India ( 2.1 million ) had the highest HIV / AIDS number of cases by
the end of 2016 . While South Africa 's large population of HIV - positive people is attributable to its high disease prevalence ( 18.9 % , one of the highest
in the world ) , Nigeria 's is lower at 2.9 % , and India 's prevalence at 0.3 % is comparable to Canada and the UK , and is lower than the United States .
</item 1>

<item 2>
List of countries by HIV/AIDS adult prevalence rate
On the other end of the spectrum , Svalbard is reported as having no cases of HIV / AIDS , while Bhutan has a much larger population but still only an estimated
246 cases through 2011 . Contents ( hide ) 1 HIV / AIDS prevalence estimates table 2 See also 3 References 4 External links HIV / AIDS prevalence estimates
table ( edit ) This data was sourced from the CIA 's world factbook unless referenced otherwise . A horizontal dash - indicates the data was not published .
Adult prevalence describes ages between 15 - 49 . Country Adult prevalence of HIV / AIDS Number of people with HIV / AIDS Annual deaths from HIV / AIDS Year of
estimate Swaziland 27.20 % 220,000 3,900 2016 Lesotho 25.00 % 330,000 9,900 2016 Botswana 21.90 % 360,000 3,900 2016 South Africa 18.90 % 7,100,000 110,000 2016
Namibia 13.80 % 230,000 4,300 2016 Zimbabwe 13.50 % 1,300,
</item 2>

<item 3>
List of countries by HIV/AIDS adult prevalence rate
AVERT . Retrieved 3 August 2017 . Jump up ^ http://factbook.lincon.com/geos/print/country/countrypdf_bf.pdf Jump up ^ `` UNAIDS.org - HIV and AIDS estimates (
2013 ) '' . Jump up ^ `` The Hindu - India has 3rd - highest number of HIV - infected people : UN '' . Jump up ^ `` The Ministry of Health has Detected ... '' .
Bhutan Observer online . 2011 - 08 - 01 . Archived from the original on 2011 - 10 - 28 . Retrieved 2011 - 11 - 21 . Jump up ^ `` CIA world factbook - country
comparison '' . Jump up ^ https://www.cdc.gov/hiv/statistics/overview/index.html Jump up ^ https://www.cdc.gov.tw/downloadfile.aspx?fid=40CE068DC7A0F111
External links ( edit ) HIV / AIDS Survey Indicators Database World Health Organisation ; Prevalence of HIV among adults aged 15 to 49 ( % ) by country World
Health Organisation ;
</item 3>

<item 4>
List of countries by HIV/AIDS adult prevalence rate
00 % 67,000 2,400 2016 Mali 1.00 % 110,000 6,100 2016 Dominican Republic 1.00 % 67,000 2,200 2016 Ukraine 0.90 % 240,000 8,500 2016 Papua New Guinea 0.90 %
46,000 1,100 2016 Cabo Verde 0.80 % 2,800 - 2016 Panama 0.80 % 21,000 - 2016 Burma 0.80 % 230,000 7,800 2016 Burkina Faso 0.80 % 95,000 3,100 2016 Latvia 0.70 %
6,600 - 2016 Congo , Democratic Republic of 0.70 % 370,000 19,000 2016 Cambodia 0.60 % 71,000 1,800 2016 Venezuela 0.60 % 120,000 2,500 2016 Moldova 0.60 %
15,000 - 2016 Eritrea 0.
</item 4>

<item 5>
List of countries by HIV/AIDS adult prevalence rate
40 % 25,000 - 2016 Costa Rica 0.40 % 13,000 - 2016 Colombia 0.40 % 120,000 2,800 2016 Vietnam 0.40 % 250,000 8,000 2016 Argentina 0.40 % 120,000 2,400 2016
Belarus 0.40 % 19,000 - 2016 USA 0.30 % 1,100,000 - 2015 Somalia 2016 Senegal 0.40 % 41,000 1,900 2016 India 0.30 % 2,100,000 62,000 2016 Ecuador 0.30 % 33,000
- 2016 Bolivia 0.30 % 19,000 - 2016 Tajikistan 0.30 % 14,000 - 2016 Peru 0.30 % 70,000 2,200 2016 Italy 0.30 % 130,000 - 2016 Laos 0.30 % 11,000 - 2016 Mexico
0.30 % 220,
</item 5>

<item 6>
List of countries by HIV/AIDS adult prevalence rate
100 2016 Jamaica 1.70 % 30,000 1,300 2016 Liberia 1.60 % 43,000 2,800 2016 Guyana 1.60 % 8,500 - 2016 Ghana 1.60 % 290,000 15,000 2016 Guinea 1.50 % 120,000
5,800 2016 Suriname 1.40 % 4,900 - 2016 Barbados 1.30 % 2,600 - 2016 Djibouti 1.30 % 8,600 - 2016 Chad 1.30 % 110,000 2,800 2016 Trinidad and Tobago 1.20 %
11,000 - 2016 Thailand 1.10 % 450,000 16,000 2016 Ethiopia 1.10 % 710,000 20,000 2016 Burundi 1.10 % 84,000 2,900 2016 Benin 1.00 % 67,000 2,400 2016 Mali 1.
</item 6>

<item 7>
List of countries by HIV/AIDS adult prevalence rate
Number of deaths due to AIDS by country World Health Organisation ; Number of people ( all ages ) living with HIV by country ( show ) HIV / AIDS topics ( show )
HIV / AIDS HIV HIV structure and genome subtypes CDC classification disease progression rates HIV / AIDS diagnosis management pathophysiology prevention
research vaccination PrEP WHO disease staging system for HIV infection and disease Children Teens / Adults Countries by AIDS prevalence rate Conditions Signs
and symptoms AIDS - defining clinical condition Diffuse infiltrative lymphocytosis syndrome Lipodystrophy Nephropathy Neurocognitive disorders Pruritus
Superinfection Tuberculosis co-infection HIV Drug Resistance Database Innate resistance to HIV History History Epidemiology Multiple sex partners Timeline AIDS
Museum Timothy Ray Brown Social AIDS orphan Catholic Church and HIV / AIDS Circumcision and HIV Criminal transmission Discrimination against people Economic
impact HIV - affected community HIV / AIDS denialism Safe sex Sex education List of HIV - positive people People With AIDS Self - Empowerment Movement Culture
International AIDS Conference International AIDS Society Joint United Nations Programme on HIV / AIDS ( UNAIDS ) President 's Emergency Plan for AIDS Relief (
PEPFAR ) Treatment Action Campaign World AIDS Day Media portrayal of HIV / AIDS Misconceptions about HIV /
</item 7>

<item 8>
List of countries by HIV/AIDS adult prevalence rate
200 - 2016 Rwanda 3.10 % 220,000 3,300 2016 Congo , Republic of the 3.10 % 91,000 3,800 2016 Guinea - Bissau 3.10 % 36,000 2,000 2016 Nigeria 2.90 % 3,200,000
160,000 2016 Cote d'Ivoire 2.70 % 460,000 25,000 2016 South Sudan 2.70 % 200,000 13,000 2016 Haiti 2.10 % 150,000 4,600 2016 Togo 2.10 % 100,000 5,100 2016
Angola 1.90 % 280,000 11,000 2016 Belize 1.80 % 4,300 - 2016 Sierra Leone 1.70 % 67,000 2,800 2016 Gambia , The 1.70 % 20,000 1,100 2016 Jamaica 1.70 % 30,000
1,
</item 8>

<item 9>
List of countries by HIV/AIDS adult prevalence rate
000 4,300 2016 Zimbabwe 13.50 % 1,300,000 30,000 2016 Zambia 12.40 % 1,200,000 21,000 2016 Mozambique 12.30 % 1,800,000 62,000 2016 Malawi 9.20 % 1,000,000
24,000 2016 Uganda 6.50 % 1,400,000 28,000 2016 Equatorial Guinea 6.20 % 35,000 - 2016 Kenya 5.40 % 1,600,000 36,000 2016 Tanzania 4.70 % 1,400,000 33,000 2016
Central African Republic 4.00 % 130,000 7,300 2016 Cameroon 3.80 % 560,000 29,000 2016 Gabon 3.60 % 48,000 1,500 2016 Bahamas , The 3.30 % 8,200 - 2016 Rwanda
3.10 % 220,000 3,
</item 9>

<item 10>
List of countries by HIV/AIDS adult prevalence rate
List of countries by HIV / AIDS adult prevalence rate - Wikipedia List of countries by HIV / AIDS adult prevalence rate Jump to : navigation , search This
article needs to be updated . Please update this article to reflect recent events or newly available information . ( October 2017 ) AIDS and HIV prevalence 2009
No data < 0.1 % 0.1 -- 0.5 % 0.5 -- 1 % 1 -- 5 % 5 -- 15 % 15 -- 50 % The human immunodeficiency virus ( HIV ) , which causes AIDS , varies in prevalence from
nation to nation . Listed here are the prevalence rates among adults in various countries , based on data from various sources , largely the CIA World Factbook
. As of 2015 , it is estimated that there are 36.7 million people worldwide infected with HIV . The HIV pandemic is most severe in South Africa . Over 10 % of
all people living with HIV / AIDS reside within the region .
</item 10>
=== END ITEM id=ca61de502d81fe8c ===

=== ITEM id=84c5ca6c5d77f976 ===
Gold sentences and facts:
Sentence 0: No . Name Field Affiliation Date of Appointment Date of Retirement Roopa Ganguly Art Bharatiya Janata Party 04 - Oct - 2016 03 - Oct - 2022 Sambhaji Raje Social work Bharatiya Janata Party 07 - Jun - 2016 03 - May - 2022 Suresh Gopi Art Bharatiya Janata Party 25 - Apr - 2016 24 - Apr - 2022 Subramanian Swamy Economics Bharatiya Janata Party 25 - Apr - 2016 24 - Apr - 2022 5 Narendra Jadhav Economics Nominated 25 - Apr - 2016 24 - Apr - 2022 6 Mary Kom Sport Nominated 25 - Apr - 2016 24 - Apr - 2022 7 Swapan Dasgupta Journalism Nominated 25 - Apr - 2016 24 - Apr - 2022 8 K.T.S. Tulsi Law Nominated 25 - Feb - 2014 24 - Feb - 2020 9 K. Parasaran Law Nominated 09 - Jun - 2012 28 - Jun - 2018 10 Rekha Art Nominated 27 - Apr - 2012 26 - Apr - 2018 11 Sachin Tendulkar Social service Nominated 27 - Apr - 2012 26 - Apr - 2018 12 Anu Aga Business Nominated 27 - Apr - 2012 26 - Apr - 2018
  Fact 0.0: Roopa Ganguly's field is Art and affiliation is Bharatiya Janata Party, with appointment on 04-Oct-2016 and retirement on 03-Oct-2022.
  Fact 0.1: Sambhaji Raje's field is Social work and affiliation is Bharatiya Janata Party, with appointment on 07-Jun-2016 and retirement on 03-May-2022.
  Fact 0.2: Suresh Gopi's field is Art and affiliation is Bharatiya Janata Party, with appointment on 25-Apr-2016 and retirement on 24-Apr-2022.
  Fact 0.3: Subramanian Swamy's field is Economics and affiliation is Bharatiya Janata Party, with appointment on 25-Apr-2016 and retirement on 24-Apr-2022.
  Fact 0.4: Narendra Jadhav (No. 5) has field Economics and affiliation Nominated, with appointment on 25-Apr-2016 and retirement on 24-Apr-2022.
  Fact 0.5: Mary Kom (No. 6) has field Sport and affiliation Nominated, with appointment on 25-Apr-2016 and retirement on 24-Apr-2022.
  Fact 0.6: Swapan Dasgupta (No. 7) has field Journalism and affiliation Nominated, with appointment on 25-Apr-2016 and retirement on 24-Apr-2022.
  Fact 0.7: K.T.S. Tulsi (No. 8) has field Law and affiliation Nominated, with appointment on 25-Feb-2014 and retirement on 24-Feb-2020.
  Fact 0.8: K. Parasaran (No. 9) has field Law and affiliation Nominated, with appointment on 09-Jun-2012 and retirement on 28-Jun-2018.
  Fact 0.9: Rekha (No. 10) has field Art and affiliation Nominated, with appointment on 27-Apr-2012 and retirement on 26-Apr-2018.
  Fact 0.10: Sachin Tendulkar (No. 11) has field Social service and affiliation Nominated, with appointment on 27-Apr-2012 and retirement on 26-Apr-2018.
  Fact 0.11: Anu Aga (No. 12) has field Business and affiliation Nominated, with appointment on 27-Apr-2012 and retirement on 26-Apr-2018.

Retrieved items:
<item 1>
List of nominated members of Rajya Sabha
Narayana Reddy 1997 -- 2003 93 Mrinal Sen 1997 -- 2003 94 -- Chaudhary Harmohan Singh Yadav 1997 -- 2003 95 -- Nanaji Deshmukh 1999 -- 2005 96 Lata Mangeshkar
1999 -- 2005 97 -- Fali Nariman 1999 -- 2005 98 -- Cho Ramaswamy 1999 -- 2005 99 Hema Malini 2003 -- 2009 100 -- Bimal Jalan 2003 -- 2009 101 K.
</item 1>

<item 2>
List of nominated members of Rajya Sabha
List of nominated members of Rajya Sabha - wikipedia List of nominated members of Rajya Sabha Jump to : navigation , search For other uses , see Member of
Parliament ( India ) . As per the Fourth Schedule ( Articles 4 ( 1 ) and 80 ( 2 ) ) of the Constitution of India , 12 members are nominated by President for
term of six years for their contributions to art , literature , science , and social services . Current nominated members of Rajya Sabha ( edit ) This is a
Current list of Members of the Rajya Sabha who have been nominated by the President . No .
</item 2>

<item 3>
List of nominated members of Rajya Sabha
Tulsi 2014 - 2020 125 Subramanian Swamy 2016 - 2022 126 MC Mary Kom 2016 - 2022 127 Narendra Jadhav 2016 - 2022 129 Suresh Gopi 2016 - 2022 130 -- Swapan
Dasgupta 2016 - 2022 131 -- Sambhaji Raje 2016 - 2022 132 -- Roopa Ganguly 2016 - 2022 References ( edit ) Nominated Members since 1952 Narender Kumar Current
list of Nominated Members Specific Jump up ^ `` Statewise List '' . 164.100. 47.5 . Retrieved 12 June 2016 . Jump up ^
http://164.100.47.5/Newmembers/partymemberlist.aspx ^ Jump up to : `` Statewise Retirement '' . 164.100. 47.5 . Retrieved 12 June 2016 .
</item 3>

<item 4>
List of nominated members of Rajya Sabha
Sr. No . Image Name Tenure -- Alladi Krishnaswamy Iyer 1952 -- 1953 Satyendranath Bose 1952 -- 1960 Prithviraj Kapoor 1952 -- 1960 -- Jagadisan Mohandas
Kumarappa 1952 -- 1954 5 -- Kalidas Nag 1952 -- 1954 6 Rukmini Devi Arundale 1952 -- 1962 7 -- N.R. Malkani 1952 -- 1962 8 -- Sahib Singh Sokhey 1952 -- 1956 9
-- Zakir Hussain 1952 -- 1962 10 -- Maithili Sharan Gupt 1952 -- 1964 11 -- Kakasaheb Kalelkar 1952 -- 1964 12 -- Radha Kumud Mukherjee 1952 -- 1958 13 P.V.
</item 4>

<item 5>
List of nominated members of Rajya Sabha
Krishnamachari 1961 -- 1967 24 -- R.R. Diwakar 1962 -- 1968 25 -- Gopal Singh 1962 -- 1968 26 -- M. Ajmal Khan 1964 -- 1972 27 -- Satyavrata Siddhantalankar
1964 -- 1968 28 Shakuntala Paranjpye 1964 -- 1970 29 -- Badri Nath Prasad 1964 -- 1970 30 -- G. Ramachandran 1964 -- 1970 31 -- M.N. Kaul 1964 -- 1972 32 --
Harivanshrai Bachchan 1966 -- 1972 33 -- Dhananjay Ramchandra Gadgil 1966 -- 1972 34 -- M.C.
</item 5>

<item 6>
List of nominated members of Rajya Sabha
No . Name Field Affiliation Date of Appointment Date of Retirement Roopa Ganguly Art Bharatiya Janata Party 04 - Oct - 2016 03 - Oct - 2022 Sambhaji Raje Social
work Bharatiya Janata Party 07 - Jun - 2016 03 - May - 2022 Suresh Gopi Art Bharatiya Janata Party 25 - Apr - 2016 24 - Apr - 2022 Subramanian Swamy Economics
Bharatiya Janata Party 25 - Apr - 2016 24 - Apr - 2022 5 Narendra Jadhav Economics Nominated 25 - Apr - 2016 24 - Apr - 2022 6 Mary Kom Sport Nominated 25 - Apr
- 2016 24 - Apr - 2022 7 Swapan Dasgupta Journalism Nominated 25 - Apr - 2016 24 - Apr - 2022 8 K.T.S.
</item 6>

<item 7>
List of nominated members of Rajya Sabha
Daphtary 1972 -- 1978 47 -- Habib Tanvir 1972 -- 1978 48 -- Krishna Kriplani 1974 -- 1980 49 Lokesh Chandra 1974 -- 1986 50 -- Scato Swu 1974 -- 1986 51 -- B.N.
Banerjee 1976 -- 1982 52 -- Malcolm Adiseshiah 1978 -- 1984 53 -- Fathema Ismail 1978 -- 1984 54 -- Pandurang Dharmaji Jadhav 1978 -- 1984 55 -- Bhagwati Charan
Varma 1978 -- 1984 56 Nargis Dutt 1980 -- 1986 57 Khushwant Singh 1980 -- 1986 58 -- Asima Chatterjee 1982 -- 1990 59 Sivaji Ganesan 1982 -- 1986 60 --
Hayatullah Ansari 1982 -- 1988 61 -- Madan Bhatia 1982 -- 1994 62 -- V.N.
</item 7>

<item 8>
List of nominated members of Rajya Sabha
Tulsi Law Nominated 25 - Feb - 2014 24 - Feb - 2020 9 K. Parasaran Law Nominated 09 - Jun - 2012 28 - Jun - 2018 10 Rekha Art Nominated 27 - Apr - 2012 26 - Apr
- 2018 11 Sachin Tendulkar Social service Nominated 27 - Apr - 2012 26 - Apr - 2018 12 Anu Aga Business Nominated 27 - Apr - 2012 26 - Apr - 2018 Complete list
of nominated members of Rajya Sabha ( edit ) This is a Complete list of Members of the Rajya Sabha who have been nominated by the President . Sr. No .
</item 8>

<item 9>
India’s Constitution provides for 12 President-nominated Rajya Sabha members serving six-year terms, chosen for contributions to fields including art,
literature, science and social services. The article lists members from 1952 onward and gives a 2016–22 cohort, including Roopa Ganguly, Sambhaji Raje, Suresh
Gopi, Subramanian Swamy, Narendra Jadhav, Mary Kom and Swapan Dasgupta. It also lists earlier nominees such as
</item 9>

<item 10>
List of nominated members of Rajya Sabha
Dutta 1993 -- 1999 85 -- Maulana Habibur Rahman Nomani 1993 -- 1999 86 Nirmala Deshpande 1997 -- 1999 and 2004 -- 2010 87 Shabana Azmi 1997 -- 2003 88 -- P.
Selvie Das 1997 -- 2003 89 -- Kartar Singh Duggal 1997 -- 2003 90 Kuldip Nayyar 1997 -- 2003 91 -- Raja Ramanna 1997 -- 2003 92 C.
</item 10>
=== END ITEM id=84c5ca6c5d77f976 ===
