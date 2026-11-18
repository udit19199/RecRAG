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

=== ITEM id=96cc20e842359566 ===
Gold sentences and facts:
Sentence 0: The Third Five - year Plan , stressed agriculture and improvement in the production of wheat , but the brief Sino - Indian War of 1962 exposed weaknesses in the economy and shifted the focus towards the defence industry and the Indian Army . In 1965 -- 1966 , India fought a War with Pakistan . There was also a severe drought in 1965 . The war led to inflation and the priority was shifted to price stabilisation . The construction of dams continued . Many cement and fertilizer plants were also built . Punjab began producing an abundance of wheat .
  Fact 0.0: The Third Five-year Plan stressed agriculture.
  Fact 0.1: The Third Five-year Plan stressed improvement in the production of wheat.
  Fact 0.2: The brief Sino-Indian War of 1962 exposed weaknesses in the economy.
  Fact 0.3: The brief Sino-Indian War of 1962 shifted the focus towards the defence industry and the Indian Army.
  Fact 0.4: India fought a war with Pakistan in 1965-1966.
  Fact 0.5: There was a severe drought in 1965.
  Fact 0.6: The war led to inflation.
  Fact 0.7: The priority was shifted to price stabilisation.
  Fact 0.8: The construction of dams continued.
  Fact 0.9: Many cement plants were built.
  Fact 0.10: Many fertilizer plants were built.
  Fact 0.11: Punjab began producing an abundance of wheat.

Retrieved items:
<item 1>
Five-Year Plans of India
Hydroelectric power projects and five steel plants at Bhilai , Durgapur , and Rourkela were established with the help of Russia , Britain ( the U.K ) and West
Germany respectively . Coal production was increased . More railway lines were added in the north east . The Tata Institute of Fundamental Research and Atomic
Energy Commission of India was established as research institutes . In 1957 a talent search and scholarship program was begun to find talented young students to
train for work in nuclear power . The total amount allocated under the Second Five - Year Plan in India was Rs. 48 billion . This amount was allocated among
various sectors : power and irrigation , social services , communications and transport , and miscellaneous . `` The target growth rate was 4.5 % and the actual
growth rate was 4.27 % . '' Third Plan ( 1961 -- 1966 ) ( edit ) The Third Five - year Plan , stressed agriculture and improvement in the production of wheat ,
but the brief Sino - Indian War of 1962 exposed weaknesses in the economy and shifted the focus towards the defence industry and the Indian Army .
</item 1>

<item 2>
Five-Year Plans of India
Contents ( hide ) 1 History 2 First Plan ( 1951 -- 1956 ) 3 Second Plan ( 1956 -- 1961 ) 4 Third Plan ( 1961 -- 1966 ) 5 Fourth Plan ( 1969 -- 1974 ) 6 Fifth
Plan ( 1974 -- 1978 ) 7 Rolling Plan ( 1978 -- 1980 ) 8 Sixth Plan ( 1980 -- 1985 ) 9 Seventh Plan ( 1985 -- 1990 ) 10 Annual Plans ( 1990 -- 1992 ) 11 Eighth
Plan ( 1992 -- 1997 ) 12 Ninth Plan ( 1997 -- 2002 ) 13 Tenth Plan ( 2002 -- 2007 ) 14 Eleventh Plan ( 2007 -- 2012 ) 15 Twelfth Plan ( 2012 -- 2017 ) 16 See
also 17 References 18 External links History ( edit ) Five - Year Plans ( FYPs ) are centralized and integrated national economic programs . Joseph Stalin
implemented the first Five Year Plan in the Soviet Union in the late 1920s .
</item 2>

<item 3>
Five-Year Plans of India
Indian Economy : Problems of development and planning . pune : Wishwa Prakashan . p. 676 . Jump up ^ `` National Development Council approves 12th Five Year
Plan '' . Indian Express . 2012 - 12 - 27 . Retrieved 2013 - 07 - 10 . Jump up ^ ( 1 ) External links ( edit ) Official website of the Planning Commission of
India India topics History Overviews Timeline Years Astronomy Economics Linguistics Maritime Mathematics Metallurgy Military Postal Science and technology
Pre-colonial Stone Age Indus Valley Civilization Vedic period Mahajanapadas Mauryas Middle kingdoms Hoysala Chola Pala Kakatiya Delhi Sultanate Vijayanagara
Mughals Marathas European trade Colonial Princely East India Company Plassey 1857 rebellion British Raj Railways Economy Army Zamindari Bengali renaissance
Political reforms Princely states Partition of Bengal Independence movement 1943 famine World War II Partition Republic Integration Non-Aligned Movement Five -
Year Plans Sino - Indian War Indo - Pakistani wars Green Revolution White Revolution Naxal Insurgency Smiling Buddha Space programme The Emergency Indian Peace
Keeping Force ( IPKF ) Economic
</item 3>

<item 4>
Five-Year Plans of India
The most important feature of this phase was active role of state in all economic sectors . Such a role was justified at that time because immediately after
independence , India was facing basic problems -- deficiency of capital and low capacity to save . The target growth rate was 2.1 % annual gross domestic
product ( GDP ) growth ; the achieved growth rate was 3.6 % the net domestic product went up by 15 % . The monsoon was good and there were relatively high crop
yields , boosting exchange reserves and the per capita income , which increased by 8 % . National income increased more than the per capita income due to rapid
population growth . Many irrigation projects were initiated during this period , including the Bhakra , Hirakud , Mettur Dam and Damodar Valley dams . The World
Health Organization ( WHO ) , with the Indian government , addressed children 's health and reduced infant mortality , indirectly contributing to population
growth . At the end of the plan period in 1956 , five Indian Institutes of Technology ( IITs ) were started as major technical institutions .
</item 4>

<item 5>
Five-Year Plans of India
Please help improve this article by adding citations to reliable sources . Unsourced material may be challenged and removed . ( June 2017 ) ( Learn how and when
to remove this template message ) The first Indian Prime Minister , Jawaharlal Nehru presented the First Five - Year Plan to the Parliament of India and needed
urgent attention . The First Five - year Plan was launched in 1951 which mainly focused in development of the primary sector . The First Five - Year Plan was
based on the Harrod -- Domar model with few modifications . The total planned budget of Rs. 2069 crore ( 2378 crore later ) was allocated to seven broad areas :
irrigation and energy ( 27.2 % ) , agriculture and community development ( 17.4 % ) , transport and communications ( 24 % ) , industry ( 8.4 % ) , social
services ( 16.6 % ) , rehabilitation of landless farmers ( 4.1 % ) , and for other sectors and services ( 2.5 % ) . The most important feature of this phase was
active role of state in all economic sectors .
</item 5>

<item 6>
Five-Year Plans of India
During 1966 -- 67 there was again the problem of drought . Equal priority was given to agriculture , its allied activities , and industrial sector . The
government of India declared `` Devaluation of Rupee '' to increase the exports of the country . The main reasons for plan holidays were the war , lack of
resources , and increase in inflation after that plan holiday was created . Fourth Plan ( 1969 -- 1974 ) ( edit ) At this time Indira Gandhi was the Prime
Minister . The Indira Gandhi government nationalised 14 major Indian banks and the Green Revolution in India advanced agriculture . In addition , the situation
in East Pakistan ( now Bangladesh ) was becoming dire as the Indo - Pakistan War of 1971 and Bangladesh Liberation War took funds earmarked for industrial
development . India also performed the Smiling Buddha underground nuclear test ( Pokhran - 1 ) in Rajasthan on May 18 , 1974 , partially in response to the
United States deployment of the Seventh Fleet in the Bay of Bengal . The fleet had been deployed to warn India against attacking West Pakistan and extending the
war .
</item 6>

<item 7>
Five-Year Plans of India
In 1965 -- 1966 , India fought a War with Pakistan . There was also a severe drought in 1965 . The war led to inflation and the priority was shifted to price
stabilisation . The construction of dams continued . Many cement and fertilizer plants were also built . Punjab began producing an abundance of wheat . Many
primary schools were started in rural areas . In an effort to bring democracy to the grass - root level , Panchayat elections were started and the states were
given more development responsibilities . State electricity boards and state secondary education boards were formed . States were made responsible for secondary
and higher education . State road transportation corporations were formed and local road building became a state responsibility . The target growth rate was 5.6
% , but the actual growth rate was 2.4 % . Due to miserable failure of the Third Plan the government was forced to declare `` plan holidays '' ( from 1966 -- 67
, 1967 -- 68 , and 1968 -- 69 ) . Three annual plans were drawn during this intervening period . During 1966 -- 67 there was again the problem of drought .
</item 7>

<item 8>
Five-Year Plans of India
The fleet had been deployed to warn India against attacking West Pakistan and extending the war . The target growth rate was 5.6 % , but the actual growth rate
was 3.3 % . Fifth Plan ( 1974 -- 1978 ) ( edit ) The Fifth Five - Year Plan laid stress on employment , poverty alleviation ( Garibi Hatao ) , and justice . The
plan also focused on self - reliance in agricultural production and defence . In 1978 the newly elected Morarji Desai government rejected the plan . The
Electricity Supply Act was amended in 1975 , which enabled the central government to enter into power generation and transmission . The Indian national highway
system was introduced and many roads were widened to accommodate the increasing traffic . Tourism also expanded . The twenty - point programme was launched in
1975 . It was followed from 1974 to 1979 . The Minimum Needs Programme ( MNP ) was introduced in the first year of the Fifth Five Year Plan ( 1974 -- 78 ) . The
objective of the programme is to provide certain basic minimum needs and thereby improve the living standards of the people .
</item 8>

<item 9>
Sino-Indian War
The U.S. nonetheless provided non-combat assistance to Indian forces and planned to send the carrier USS Kitty Hawk to the Bay of Bengal to support India in
case of an air war . Some reports suggest a contradictory response from the U.S. According to former Indian diplomat G. Parthasarathy , `` only after we got
nothing from the US did arms supplies from the Soviet Union to India commence . '' In 1962 , President of Pakistan Ayub Khan made clear to India that Indian
troops could safely be transferred from the Pakistan frontier to the Himalayas . Aftermath ( edit ) China ( edit ) According to the China 's official military
history , the war achieved China 's policy objectives of securing borders in its western sector , as China retained de facto control of the Aksai Chin . After
the war , India abandoned the Forward Policy , and the de facto borders stabilised along the Line of Actual Control . According to James Calvin of Marine Corps
Command and Staff College , even though China won a military victory it lost in terms of its international image .
</item 9>

<item 10>
Five-Year Plans of India
Five - Year plans of India - Wikipedia Five - Year plans of India Jump to : navigation , search From 1947 to 2017 , the Indian economy was premised on the
concept of planning . This was carried through the Five - Year Plans , developed , executed , and monitored by the Planning Commission ( 1951 - 2014 ) and the
NITI Aayog ( 2015 - 2017 ) . With the prime minister as the ex-officio chairman , the commission has a nominated deputy chairman , who holds the rank of a
cabinet Minister . Montek Singh Ahluwalia is the last deputy chairman of the commission ( resigned on 26 May 2014 ) . The Twelfth Plan completed its term in
March 2017 . Prior to the Fourth Plan , the allocation of state resources was based on schematic patterns rather than a transparent and objective mechanism ,
which led to the adoption of the Gadgil formula in 1969 . Revised versions of the formula have been used since then to determine the allocation of central
assistance for state plans .
</item 10>
=== END ITEM id=96cc20e842359566 ===

=== ITEM id=bd411a26c0927d9a ===
Gold sentences and facts:
Sentence 0: When leaving work in Idaho 's Snake River Valley , Billy Colman sees a pack of dogs attacking a stray coonhound . He takes the stray home to feed it . Once it has rested , Billy sets it free , knowing that it will return home .
  Fact 0.0: Billy Colman leaves work in Idaho's Snake River Valley.
  Fact 0.1: Billy Colman sees a pack of dogs attacking a stray coonhound.
  Fact 0.2: Billy Colman takes the stray coonhound home to feed it.
  Fact 0.3: After the coonhound has rested, Billy Colman sets it free.
  Fact 0.4: Billy Colman knows that the coonhound will return home.

Retrieved items:
<item 1>
Where the Red Fern Grows
On the way back home , he spends the night in Robber 's Cave in the Sparrow Hawk Mountains . There he builds a fire and plays with the puppies . While trying to
sleep , he hears a noise that he realizes is the cry of a mountain lion . In the morning , he continues on . He comes to a sycamore tree and sees the names Dan
and Ann carved inside a heart in the bark and decides to name the puppies Old Dan and Little Ann . To train Old Dan and Little Ann , Billy traps a raccoon with
the help of his grandfather and uses its pelt to teach them to hunt . During their training , their personalities become apparent : Old Dan is brave and strong
, while Little Ann is very intelligent . Both are very loyal to each other and to Billy . On the first night of hunting season , Billy takes Old Dan and Little
Ann out for their first hunt . He promises them that if they tree a raccoon , he will do the rest . They tree one in a large sycamore , which Billy had
previously nicknamed `` The Big Tree '' .
</item 1>

<item 2>
Where the Red Fern Grows
As he tries to call them off , they look at him sadly and he cuts the tree down , which takes him two days and costs him blistered hands . In the end , when he
's about to give up his effort , Billy offers a short prayer for strength to continue . A strong wind starts to blow and the tree falls . Old Dan and Little Ann
take the raccoon down . Billy , Old Dan , and Little Ann go hunting almost every night and become well - known in the Ozarks . Billy and his grandfather make a
bet with Rubin and Rainie Pritchard that Old Dan and Little Ann can tree the `` ghost coon '' . It leads them on a long , complicated hunt , and Rubin and
Rainie want to give up . When they finally have it treed , Billy refuses to kill it . Rubin then orders his dog , Old Blue , to attack Old Dan , who is smaller
; and Rubin starts to beat Billy . Little Ann helps Old Dan , and together they injure Old Blue ; Rubin runs to attack them with an axe , but he trips , falls
on it and kills himself .
</item 2>

<item 3>
Where the Red Fern Grows
Reception ( edit ) Although sales of the novel began slowly , by 1974 over 90,000 copies had been sold . In 2001 , Publishers Weekly estimated that it had sold
6,754,308 copies . Today Where the Red Fern Grows is required reading in many American schools . One critic said it will please adults as well as children . I
remember crying so much through this book , and even today I tear up thinking of Old Dan and Little Ann . I also loaned this to my ( then ) children 's
librarian , because the library copy was always out . I even marked the pages , `` Get out tissue here . '' -- DeAnn Okamura I love , love , love this book with
all my heart and soul . My fourth grade teacher read it to me eons ago , and I 've read it to two of my three boys . There 's something about weeping together
uncontrollably that builds a community of readers ... -- Tess Alfonsin There is a statue of Billy and his dogs at the Idaho Falls Public Library . Every year in
Tahlequah Oklahoma there is a Red Fern festival .
</item 3>

<item 4>
Where the Red Fern Grows
Where the Red Fern Grows - wikipedia Where the Red Fern Grows Jump to : navigation , search For the film adaptations , see Where the Red Fern Grows ( 1974 film
) and Where the Red Fern Grows ( 2003 film ) . Where the Red Fern Grows First edition hardback cover Author Wilson Rawls Country United States Language English
Genre Children 's novel Publisher Doubleday Publication date 1961 Media type Print ( Hardcover ) Pages 245 pp ISBN 0 - 440 - 22814 - X OCLC 39850615 Where the
Red Fern Grows is a 1961 children 's novel by Wilson Rawls about a boy who buys and trains two Redbone Coonhound hunting dogs . Contents ( hide ) 1 Plot summary
2 Films 3 Background 4 Reception 5 See also 6 Footnotes 7 External links Plot summary ( edit ) When leaving work in Idaho 's Snake River Valley , Billy Colman
sees a pack of dogs attacking a stray coonhound . He takes the stray home to feed it . Once it has rested , Billy sets it free , knowing that it will return
home .
</item 4>

<item 5>
Where the Red Fern Grows
Upon the second submission to the Saturday Evening Post , it was accepted . It was first published in serialization in the Saturday Evening Post in 1961 under
the title The Hounds of Youth . Doubleday then accepted the book for publication . Rawls said Doubleday then `` broke my heart . '' They changed the title to
Where the Red Fern Grows , and attempted to market it to adult readers . For about six years , it languished on shelves and failed to sell . Doubleday was going
to put it out of print , but one agent named Mr. Breinholt from Salt Lake City fought for it and asked for just a few more months to market it . He got Rawls a
speaking engagement at the University of Utah to a conference of over 5,000 reading teachers and librarians . Copies of it were made available to them . When
they took it back to their schools , the children loved it , and orders began pouring in . Jim Trelease states , `` Each year since then , it has sold more
copies than the previous year . ''
</item 5>

<item 6>
Where the Red Fern Grows
Every year in Tahlequah Oklahoma there is a Red Fern festival . It is listed below in the links . See also ( edit ) List of Where the Red Fern Grows Characters
Footnotes ( edit ) Jump up ^ Where the Red Fern Grows Discussion Guide Scholastic.com ^ Jump up to : Where the Red Fern Grows ( 1974 ) on IMDb Jump up ^ IMDb :
Where the Red Fern Grows : Part Two ^ Jump up to : Wilson Rawls , Dreams Can Come True , audio recording with commentary by Jim Trelease ^ Jump up to :
http://www.trelease-on-reading.com/rawls.html Jump up ^
https://news.google.com/newspapers?nid=336&dat=19740216&id=bMdSAAAAIBAJ&sjid=AH4DAAAAIBAJ&pg=5192,3528282&hl=en Jump up ^ `` All - Time Bestselling Children 's
Books '' . PublishersWeekly.com . Retrieved 2017 - 01 - 29 .
</item 6>

<item 7>
Where the Red Fern Grows
Billy 's grandfather enters him into a championship coon hunt , putting him against experienced hunters and the finest dogs in the country . Before it starts ,
he enters Little Ann into a conformation contest , where she wins the silver cup . On the fourth night of the hunt , Old Dan and Little Ann tree three raccoons
, making it to the final round . The sixth night , they tree one before a blizzard hits . Billy , his father , grandfather , and the judge lose sight of the
dogs . When they finally find them , Billy 's grandfather sprains his ankle . They build a fire , and when Billy 's father chops down a tree , three raccoons
jump out . Old Dan and Little Ann take two of them down , and chase the third to another tree . In the morning , the hunters find them covered with ice circling
the tree . That raccoon wins them the championship , gold cup , and $300 of jackpot money . One night Old Dan and Little Ann tree a mountain lion , and it
attacks . Billy enters the fight with his axe , hoping to save his dogs , but they end up having to save him .
</item 7>

<item 8>
Where the Red Fern Grows
Once it has rested , Billy sets it free , knowing that it will return home . The experience reminds Billy of his childhood in the Ozark Mountains of Oklahoma .
He wants a dog , and his parents offer to get him a mongrel from a neighbor . He wants two coonhounds , but his parents tell him they can not afford them .
Billy finds a magazine ad for a Kentucky kennel which breeds Redbone Coonhounds and sells them for $25 each . He goes to work performing odd jobs and saves the
$50 he needs with the help of his grandfather . Since it takes two years to do so , his grandfather writes to see if the kennel will honor the ad . They do ,
and the price has dropped so the two puppies only cost $40 . Billy 's dogs are delivered to the freight depot in Tahlequah . His family does not have
transportation , so he finds his own way to get to the depot and walks through the hills . He picks up his puppies , which are a male and a female . He buys a
gift for each member of his family with his extra $10 before heading back home .
</item 8>

<item 9>
Where the Red Fern Grows
PublishersWeekly.com . Retrieved 2017 - 01 - 29 . Jump up ^
https://news.google.com/newspapers?nid=336&dat=19730929&id=3dBSAAAAIBAJ&sjid=G38DAAAAIBAJ&pg=6645,6875378&hl=en Jump up ^ `` The Catoosa County News - Google
News Archive Search '' . news.google.com . Retrieved 2017 - 01 - 29 . Jump up ^ `` Top 100 Children 's Novels # 34 : Where the Red Fern Grows by Wilson Rawls ''
. School Library Journal . Retrieved 2017 - 02 - 10 . Jump up ^ `` Wilson Rawls '' . Idaho Falls Public Library . Retrieved 2017 - 02 - 10 . External links (
edit ) Oracle Think Quest , Education Foundation : Where the Red Fern Grows . SparkNotes : Where the Red Fern Grows .
</item 9>

<item 10>
Where the Red Fern Grows
Eventually , they kill the mountain lion , but Old Dan is badly wounded , and Billy finds his intestines in a bush . He dies late that night . Billy is
heartbroken , and Little Ann loses the will to live , stops eating , and dies of grief a few days later on Old Dan 's grave . Billy 's father tries to tell him
that it is all for the best , because with the money they received from winning the championship hunt , they can move to town . He goes to visit Old Dan and
Little Ann 's graves and finds a giant red fern between them . According to Native American legend , only an angel can plant one . He feels ready to move on
knowing that his dogs are always going to be remembered . Films ( edit ) The novel was made into a 1974 film starring Stewart Petersen , James Whitmore ,
Beverly Garland , and Jack Ging . It was followed by a sequel in 1992 , which starred Wilford Brimley , Chad McQueen , Lisa Whelchel , and Karen Carlson . The
film was remade in 2003 and starred Joseph Ashton , Dabney Coleman , Ned Beatty and Dave Matthews .
</item 10>
=== END ITEM id=bd411a26c0927d9a ===

=== ITEM id=0c13b76637b8aa0f ===
Gold sentences and facts:
Sentence 0: Mac OS X , OS X , and macOS version information Version Codename Darwin version Processor support Application support Kernel Date announced Release date Most recent version Rhapsody Developer Release Grail1Z4 / Titan1U 32 - bit PowerPC 32 - bit PowerPC 32 - bit Unknown August 31 , 1997 DR2 ( May 14 , 1998 ) Mac OS X Server 1.0 Hera Unknown March 16 , 1999 1.2 v3 ( October 27 , 2000 ) Mac OS X Developer Preview Unknown May 11 , 1998 March 16 , 1999 DP4 ( April 5 , 2000 ) Mac OS X Public Beta Kodiak 1.2. 1 Unknown September 13 , 2000 N / A Mac OS X 10.0 Cheetah 1.3. 1 Unknown March 24 , 2001 10.0. 4 ( June 22 , 2001 ) Mac OS X 10.1 Puma 1.4. 1 / 5 July 18 , 2001 September 25 , 2001 10.1. 5 ( June 6 , 2002 ) Mac OS X 10.2 Jaguar 6 32 / 64 - bit PowerPC May 6 , 2002 August 24 , 2002 10.2. 8 ( October 3 , 2003 ) Mac OS X 10.3 Panther 7 32 / 64 - bit PowerPC June 23 , 2003 October 24 , 2003 10.3. 9 ( April 15 , 2005 ) Mac OS X 10.4 Tiger 8 32 / 64 - bit PowerPC and Intel 32 / 64 - bit PowerPC and Intel May 4 , 2004 April 29 , 2005 10.4. 11 ( November 14 , 2007 ) Mac OS X 10.5 Leopard 9 32 / 64 - bit PowerPC and Intel June 26 , 2006 October 26 , 2007 10.5. 8 ( August 5 , 2009 ) Mac OS X 10.6 Snow Leopard 10 32 / 64 - bit Intel 32 / 64 - bit Intel 32 - bit PowerPC 32 / 64 - bit June 9 , 2008 August 28 , 2009 10.6. 8 v1. 1 ( July 25 , 2011 ) Mac OS X 10.7 Lion 11 64 - bit Intel 32 / 64 - bit Intel October 20 , 2010 July 20 , 2011 10.7. 5 ( September 19 , 2012 ) OS X 10.8 Mountain Lion 12 64 - bit February 16 , 2012 July 25 , 2012 10.8. 5 ( 12F45 ) ( October 3 , 2013 ) OS X 10.9 Mavericks 13 June 10 , 2013 October 22 , 2013 10.9. 5 ( 13F1112 ) ( September 18 , 2014 ) OS X 10.10 Yosemite 14 June 2 , 2014 October 16 , 2014 10.10. 5 ( 14F27 ) ( August 13 , 2015 ) OS X 10.11 El Capitan 15 June 8 , 2015 September 30 , 2015 10.11. 6 ( 15G31 ) ( July 18 , 2016 ) macOS 10.12 Sierra 16 June 13 , 2016 September 20 , 2016 10.12. 6 ( 16G29 ) ( July 19 , 2017 ) macOS 10.13 High Sierra 17 June 5 , 2017 September 25 , 2017 10.13. 4 ( 17E199 ) ( March 29 , 2018 )
  Fact 0.0: Rhapsody Developer Release had the codenames Grail1Z4 and Titan1U
  Fact 0.1: Rhapsody Developer Release supported 32-bit PowerPC processors
  Fact 0.2: Rhapsody Developer Release applications were 32-bit PowerPC
  Fact 0.3: Rhapsody Developer Release was announced August 31, 1997
  Fact 0.4: The most recent Rhapsody Developer Release version is DR2 (May 14, 1998)
  Fact 0.5: Mac OS X Server 1.0 had the codename Hera
  Fact 0.6: Mac OS X Server 1.0 was released March 16, 1999
  Fact 0.7: The most recent Mac OS X Server 1.0 version is 1.2v3 (October 27, 2000)
  Fact 0.8: Mac OS X Developer Preview was announced May 11, 1998
  Fact 0.9: The most recent Mac OS X Developer Preview version is DP4 (April 5, 2000)
  Fact 0.10: Mac OS X Public Beta had the codename Kodiak
  Fact 0.11: Mac OS X Public Beta had Darwin version 1.2.1
  Fact 0.12: Mac OS X Public Beta was released September 13, 2000
  Fact 0.13: Mac OS X Public Beta has no most recent version listed (N/A)
  Fact 0.14: Mac OS X 10.0 had the codename Cheetah
  Fact 0.15: Mac OS X 10.0 had Darwin version 1.3.1
  Fact 0.16: Mac OS X 10.0 was released March 24, 2001
  Fact 0.17: The most recent Mac OS X 10.0 version is 10.0.4 (June 22, 2001)
  Fact 0.18: Mac OS X 10.1 had the codename Puma
  Fact 0.19: Mac OS X 10.1 had Darwin version 1.4.1 / 5
  Fact 0.20: Mac OS X 10.1 was announced July 18, 2001
  Fact 0.21: Mac OS X 10.1 was released September 25, 2001
  Fact 0.22: The most recent Mac OS X 10.1 version is 10.1.5 (June 6, 2002)
  Fact 0.23: Mac OS X 10.2 had the codename Jaguar
  Fact 0.24: Mac OS X 10.2 had Darwin version 6
  Fact 0.25: Mac OS X 10.2 supported 32-bit and 64-bit PowerPC processors
  Fact 0.26: Mac OS X 10.2 was announced May 6, 2002
  Fact 0.27: Mac OS X 10.2 was released August 24, 2002
  Fact 0.28: The most recent Mac OS X 10.2 version is 10.2.8 (October 3, 2003)
  Fact 0.29: Mac OS X 10.3 had the codename Panther
  Fact 0.30: Mac OS X 10.3 had Darwin version 7
  Fact 0.31: Mac OS X 10.3 supported 32-bit and 64-bit PowerPC processors
  Fact 0.32: Mac OS X 10.3 was announced June 23, 2003
  Fact 0.33: Mac OS X 10.3 was released October 24, 2003
  Fact 0.34: The most recent Mac OS X 10.3 version is 10.3.9 (April 15, 2005)
  Fact 0.35: Mac OS X 10.4 had the codename Tiger
  Fact 0.36: Mac OS X 10.4 had Darwin version 8
  Fact 0.37: Mac OS X 10.4 supported 32-bit and 64-bit PowerPC and Intel processors
  Fact 0.38: Mac OS X 10.4 was announced May 4, 2004
  Fact 0.39: Mac OS X 10.4 was released April 29, 2005
  Fact 0.40: The most recent Mac OS X 10.4 version is 10.4.11 (November 14, 2007)
  Fact 0.41: Mac OS X 10.5 had the codename Leopard
  Fact 0.42: Mac OS X 10.5 had Darwin version 9
  Fact 0.43: Mac OS X 10.5 supported 32-bit and 64-bit PowerPC and Intel processors
  Fact 0.44: Mac OS X 10.5 was announced June 26, 2006
  Fact 0.45: Mac OS X 10.5 was released October 26, 2007
  Fact 0.46: The most recent Mac OS X 10.5 version is 10.5.8 (August 5, 2009)
  Fact 0.47: Mac OS X 10.6 had the codename Snow Leopard
  Fact 0.48: Mac OS X 10.6 had Darwin version 10
  Fact 0.49: Mac OS X 10.6 supported 32-bit and 64-bit Intel processors
  Fact 0.50: Mac OS X 10.6 supported 32-bit PowerPC processors
  Fact 0.51: Mac OS X 10.6 was announced June 9, 2008
  Fact 0.52: Mac OS X 10.6 was released August 28, 2009
  Fact 0.53: The most recent Mac OS X 10.6 version is 10.6.8 v1.1 (July 25, 2011)
  Fact 0.54: Mac OS X 10.7 had the codename Lion
  Fact 0.55: Mac OS X 10.7 had Darwin version 11
  Fact 0.56: Mac OS X 10.7 supported 64-bit Intel processors
  Fact 0.57: Mac OS X 10.7 was announced October 20, 2010
  Fact 0.58: Mac OS X 10.7 was released July 20, 2011
  Fact 0.59: The most recent Mac OS X 10.7 version is 10.7.5 (September 19, 2012)
  Fact 0.60: OS X 10.8 had the codename Mountain Lion
  Fact 0.61: OS X 10.8 had Darwin version 12
  Fact 0.62: OS X 10.8 supported 64-bit Intel processors
  Fact 0.63: OS X 10.8 was announced February 16, 2012
  Fact 0.64: OS X 10.8 was released July 25, 2012
  Fact 0.65: The most recent OS X 10.8 version is 10.8.5 (12F45) (October 3, 2013)
  Fact 0.66: OS X 10.9 had the codename Mavericks
  Fact 0.67: OS X 10.9 had Darwin version 13
  Fact 0.68: OS X 10.9 was announced June 10, 2013
  Fact 0.69: OS X 10.9 was released October 22, 2013
  Fact 0.70: The most recent OS X 10.9 version is 10.9.5 (13F1112) (September 18, 2014)
  Fact 0.71: OS X 10.10 had the codename Yosemite
  Fact 0.72: OS X 10.10 had Darwin version 14
  Fact 0.73: OS X 10.10 was announced June 2, 2014
  Fact 0.74: OS X 10.10 was released October 16, 2014
  Fact 0.75: The most recent OS X 10.10 version is 10.10.5 (14F27) (August 13, 2015)
  Fact 0.76: OS X 10.11 had the codename El Capitan
  Fact 0.77: OS X 10.11 had Darwin version 15
  Fact 0.78: OS X 10.11 was announced June 8, 2015
  Fact 0.79: OS X 10.11 was released September 30, 2015
  Fact 0.80: The most recent OS X 10.11 version is 10.11.6 (15G31) (July 18, 2016)
  Fact 0.81: macOS 10.12 had the codename Sierra
  Fact 0.82: macOS 10.12 had Darwin version 16
  Fact 0.83: macOS 10.12 was announced June 13, 2016
  Fact 0.84: macOS 10.12 was released September 20, 2016
  Fact 0.85: The most recent macOS 10.12 version is 10.12.6 (16G29) (July 19, 2017)
  Fact 0.86: macOS 10.13 had the codename High Sierra
  Fact 0.87: macOS 10.13 had Darwin version 17
  Fact 0.88: macOS 10.13 was announced June 5, 2017
  Fact 0.89: macOS 10.13 was released September 25, 2017
  Fact 0.90: The most recent macOS 10.13 version is 10.13.4 (17E199) (March 29, 2018)
Sentence 1: macOS Screenshot of the macOS High Sierra desktop Developer Apple , Inc . Written in C++ Objective - C Swift OS family Macintosh , Unix Working state Current Source model Closed source ( with open source components ) Initial release March 24 , 2001 ; 17 years ago ( 2001 - 03 - 24 ) Latest release 10.13. 4 ( 17E202 ) ( April 24 , 2018 ; 2 days ago ( 2018 - 04 - 24 ) ) ( ± ) Latest preview 10.13. 5 beta 2 ( 17B45c ) ( April 17 , 2018 ; 9 days ago ( 2018 - 04 - 17 ) ) ( ± ) Marketing target Personal computing Available in 34 languages List of languages As of macOS Sierra : Arabic , Catalan , Croatian , Chinese ( Simplified ) , Chinese ( Traditional ) , Czech , Danish , Dutch , English , Finnish , French , German , Greek , Hebrew , Hungarian , Indonesian , Italian , Japanese , Korean , Malay , Norwegian , Polish , Portuguese ( Brazil ) , Portuguese ( Portugal ) , Romanian , Russian , Slovak , Spanish ( Mexico ) , Spanish ( Spain ) , Swedish , Thai , Turkish , Ukrainian , Vietnamese Update method Mac App Store ( 10.8 + ) Software Update ( 10.0 -- 10.7 ) Platforms x86 - 64 ( 10.4. 7 - present ) IA - 32 ( 10.4. 4 -- 10.6. 8 ; discontinued ) PowerPC ( 10.0 -- 10.5. 8 ; discontinued ) Kernel type Hybrid ( XNU ) Default user interface Graphical ( Aqua ) License Commercial software , proprietary software Preceded by Classic Mac OS Official website www.apple.com/macos Support status Supported
  Fact 1.0: macOS infobox: developer is Apple Inc.
  Fact 1.1: macOS is written in C++, Objective-C, and Swift
  Fact 1.2: macOS belongs to the Macintosh and Unix OS family
  Fact 1.3: macOS is a current working system
  Fact 1.4: macOS is closed source, with open source components
  Fact 1.5: macOS initial release was March 24, 2001
  Fact 1.6: Latest macOS release is 10.13.4 (17E202), April 24, 2018
  Fact 1.7: Latest macOS preview is 10.13.5 beta 2 (17B45c), April 17, 2018
  Fact 1.8: macOS marketing target is personal computing
  Fact 1.9: As of macOS Sierra, macOS is available in 34 languages: Arabic, Catalan, Croatian, Chinese (Simplified), Chinese (Traditional), Czech, Danish, Dutch, English, Finnish, French, German, Greek, Hebrew, Hungarian, Indonesian, Italian, Japanese, Korean, Malay, Norwegian, Polish, Portuguese (Brazil), Portuguese (Portugal), Romanian, Russian, Slovak, Spanish (Mexico), Spanish (Spain), Swedish, Thai, Turkish, Ukrainian, and Vietnamese
  Fact 1.10: macOS is updated through the Mac App Store for 10.8 and later
  Fact 1.11: macOS 10.0 to 10.7 were updated through Software Update
  Fact 1.12: macOS runs on x86-64 from 10.4.7 to present
  Fact 1.13: macOS ran on IA-32 from 10.4.4 to 10.6.8, now discontinued
  Fact 1.14: macOS ran on PowerPC from 10.0 to 10.5.8, now discontinued
  Fact 1.15: macOS kernel type is hybrid (XNU)
  Fact 1.16: macOS default user interface is graphical (Aqua)
  Fact 1.17: macOS license is commercial and proprietary software
  Fact 1.18: macOS is preceded by Classic Mac OS
  Fact 1.19: macOS official website is www.apple.com/macos
  Fact 1.20: macOS support status is supported
Sentence 2: macOS is the second major series of Macintosh operating systems . The first is colloquially called the `` classic '' Mac OS , which was introduced in 1984 , and the final release of which was Mac OS 9 in 1999 . The first desktop version , Mac OS X 10.0 , was released in March 2001 , with its first update , 10.1 , arriving later that year . After this , Apple began naming its releases after big cats , which lasted until OS X 10.8 Mountain Lion . Since OS X 10.9 Mavericks , releases have been named after landmarks in California . Apple shortened the name to `` OS X '' in 2012 and then changed it to `` macOS '' in 2016 , adopting the nomenclature that it uses for their other operating systems , iOS , watchOS , and tvOS . The latest version is macOS High Sierra , which was publicly released in September 2017 .
  Fact 2.0: macOS is the second major series of Macintosh operating systems
  Fact 2.1: The first series is colloquially called the classic Mac OS
  Fact 2.2: Classic Mac OS was introduced in 1984
  Fact 2.3: The final classic Mac OS release was Mac OS 9 in 1999
  Fact 2.4: The first desktop version of macOS was Mac OS X 10.0, released in March 2001
  Fact 2.5: Mac OS X 10.1 was the first update, arriving later in 2001
  Fact 2.6: After this, Apple named its releases after big cats, until OS X 10.8 Mountain Lion
  Fact 2.7: Since OS X 10.9 Mavericks, releases have been named after landmarks in California
  Fact 2.8: Apple shortened the name to OS X in 2012
  Fact 2.9: Apple changed the name to macOS in 2016
  Fact 2.10: The name change adopted the nomenclature Apple uses for iOS, watchOS, and tvOS
  Fact 2.11: The latest version is macOS High Sierra
  Fact 2.12: macOS High Sierra was publicly released in September 2017

Retrieved items:
<item 1>
macOS
8 ; discontinued ) PowerPC ( 10.0 -- 10.5. 8 ; discontinued ) Kernel type Hybrid ( XNU ) Default user interface Graphical ( Aqua ) License Commercial software ,
proprietary software Preceded by Classic Mac OS Official website www.apple.com/macos Support status Supported macOS ( / ˌmækoʊˈɛs / ; previously Mac OS X , then
OS X ) is a series of graphical operating systems developed and marketed by Apple Inc. since 2001 . It is the primary operating system for Apple 's Mac family
of computers . Within the market of desktop , laptop and home computers , and by web usage , it is the second most widely used desktop OS , after Microsoft
Windows . macOS is the second major series of Macintosh operating systems . The first is colloquially called the `` classic '' Mac OS , which was introduced in
1984 , and the final release of which was Mac OS 9 in 1999 . The first desktop version , Mac OS X 10.0 , was released in March 2001 , with its first update ,
10.1 , arriving later that year .
</item 1>

<item 2>
macOS
Apple 's 2015 update , OS X 10.11 El Capitan , was announced to focus specifically on stability and performance improvements . MacOS In 2016 , with the release
of macOS 10.12 Sierra , the name was changed from OS X to macOS to streamline it with the branding of Apple 's other primary operating systems : iOS , watchOS ,
and tvOS . macOS 10.12 Sierra 's main features are the introduction of Siri to macOS , Optimized Storage , improvements to included applications , and greater
integration with Apple 's iPhone and Apple Watch . The Apple File System ( APFS ) was announced at the Apple Worldwide Developers Conference in 2016 as a
replacement for HFS+ , a highly criticized file system . At the 2017 Worldwide Developers Conference Apple previewed macOS 10.13 High Sierra . It uses APFS ,
rather than HFS+ , on solid state drives . Architecture Main article : Architecture of macOS At macOS 's core is a POSIX compliant operating system built on top
of the XNU kernel , with standard Unix facilities available from the command line interface . Apple has released this family of software as a free and open
source operating system named Darwin .
</item 2>

<item 3>
macOS
January 21 , 2009 . Retrieved January 29 , 2009 . Jump up ^ `` XProtect Explained : How Your Mac 's Built - in Anti-malware Software Works '' . How - To Geek .
May 18 , 2015 . Jump up ^ Nudd , Tim . `` Apple 's Get a Mac campaign '' . AdWeek . Retrieved 2 December 2015 . Jump up ^ Arthur , Charles . `` Apple tweaks
Microsoft over Vista ad spending '' . The Guardian . Retrieved 2 December 2015 . External links macOS -- official site macOS History Architecture Components
Technologies Server Software Versions Server 1.0 Hera Public Beta Kodiak 10.0 Cheetah 10.1 Puma 10.2 Jaguar 10.3 Panther 10.4 Tiger 10.5 Leopard 10.6 Snow
Leopard 10.7 Lion 10.8 Mountain Lion 10.9 Mavericks 10.10 Yosemite 10.11 El Capitan 10.12 Sierra 10.13 High Sierra Applications Automator Calculator Calendar
Chess Contacts Dashboard Dictionary DVD Player FaceTime Finder Game Center Grapher iTunes ( version history )
</item 3>

<item 4>
macOS
Left to right : Cheetah / Puma ( 1 ) , Jaguar ( 2 ) , Panther ( 3 ) , Tiger ( 4 ) , Leopard ( 5 ) , Snow Leopard ( 6 ) , Lion ( 7 ) , Mountain Lion ( 8 ) ,
Mavericks ( 9 ) , Yosemite ( 10 ) , El Capitan ( 11 ) , Sierra ( 12 ) High Sierra ( 13 ) . Mac OS X , OS X , and macOS version information Version Codename
Darwin version Processor support Application support Kernel Date announced Release date Most recent version Rhapsody Developer Release Grail1Z4 / Titan1U 32 -
bit PowerPC 32 - bit PowerPC 32 - bit Unknown August 31 , 1997 DR2 ( May 14 , 1998 ) Mac OS X Server 1.0 Hera Unknown March 16 , 1999 1.2 v3 ( October 27 , 2000
) Mac OS X Developer Preview Unknown May 11 , 1998 March 16 , 1999 DP4 ( April 5 , 2000 ) Mac OS X Public Beta Kodiak 1.2.
</item 4>

<item 5>
macOS
Similar to Mac OS X 10.6 Snow Leopard , Apple described this release as containing `` refinements to the Mac experience '' and `` improvements to system
performance '' rather than new features . Refinements include public transport built into the Maps application , GUI improvements to the Notes application ,
adopting San Francisco as the system font for clearer legibility , and the introduction of System Integrity Protection . The Metal API , first introduced in iOS
8 , was also included in this operating system for `` all Macs since 2012 '' . MacOS 10.12 Sierra Main article : macOS Sierra macOS 10.12 Sierra was released to
the public on September 20 , 2016 . New features include the addition of Siri , Optimized Storage , and updates to Photos , Messages , and iTunes . MacOS 10.13.
High Sierra Main article : macOS High Sierra The fall release of macOS 10.13 High Sierra was announced on June 5 , 2017 at Apple 's WWDC event . On September 12
, 2017 at Apple 's iPhone event , they announced its release to be September 25 , 2017 .
</item 5>

<item 6>
macOS
5 ( 12F45 ) ( October 3 , 2013 ) OS X 10.9 Mavericks 13 June 10 , 2013 October 22 , 2013 10.9. 5 ( 13F1112 ) ( September 18 , 2014 ) OS X 10.10 Yosemite 14 June
2 , 2014 October 16 , 2014 10.10. 5 ( 14F27 ) ( August 13 , 2015 ) OS X 10.11 El Capitan 15 June 8 , 2015 September 30 , 2015 10.11. 6 ( 15G31 ) ( July 18 ,
2016 ) macOS 10.12 Sierra 16 June 13 , 2016 September 20 , 2016 10.12. 6 ( 16G29 ) ( July 19 , 2017 ) macOS 10.13 High Sierra 17 June 5 , 2017 September 25 ,
2017 10.13.
</item 6>

<item 7>
macOS
Folders Speakable items Spotlight Stacks System Integrity Protection Uniform Type Identifier Universal binary WebKit XNU XQuartz Deprecated Carbon HFS+
Discontinued BootX Brushed metal Classic Environment Rosetta Spaces Xgrid Links to related articles Darwin - derived operating systems History Pre-Darwin Unix
BSD Mach NeXTSTEP OpenStep GNUstep Rhapsody Macintosh Mac OS X / OS X / macOS ( History ) Mac OS X Server Mac OS X Server 1.0 ( Hera / Rhapsody ) 10.0 ( Cheetah
) 10.1 ( Puma ) 10.2 ( Jaguar ) 10.3 ( Panther ) 10.4 ( Tiger ) 10.5 Leopard Server 10.6 Snow Leopard Server Mac OS X Public Beta ( Kodiak ) 10.0 ( Cheetah )
10.1 ( Puma ) 10.2 ( Jaguar ) 10.3 Panther 10.4 Tiger 10.5 Leopard 10.6 Snow Leopard 10.7 Lion OS X 10.8 Mountain Lion 10.9 Mavericks 10.10 Yosemite 10.11 El
Capitan macOS 10.12 Sierra 10.
</item 7>

<item 8>
macOS
MacRumors . Jump up ^ `` macOS High Sierra 10.13. 4 ( 17E202 ) - Releases '' . Apple Developer . Apple Inc . April 24 , 2018 . Jump up ^ Juli Clover ( April 17
, 2018 ) . `` Apple Seeds Second Beta of macOS High Sierra 10.13. 5 to Developers '' . MacRumors . Retrieved April 18 , 2018 . Jump up ^ `` macOS High Sierra
10.13. 5 beta ( 17B45c ) - Releases - Apple Developer '' . developer.apple.com. April 17 , 2018 . Retrieved April 18 , 2018 . ^ Jump up to : `` macOS -- How to
Upgrade -- Apple '' . Apple . Retrieved September 28 , 2016 . Jump up ^ Apple Events - WWDC Keynote June 2016 . Event occurs at 36 : 28 . Jump up ^ `` Desktop
Operating System Market Share '' . Net Applications . Jump up ^ `` Top 8 Operating Systems from Sept 2011 to Aug 2015 '' .
</item 8>

<item 9>
macOS
12 OS X 10.10 Yosemite 4.13 OS X 10.11 El Capitan 4.14 macOS 10.12 Sierra 4.15 macOS 10.13 High Sierra 5 Reception 5.1 Usage share 5.2 Malware and spyware 5.3
Promotion 6 See also 7 References 8 External links History Development Simplified history of Unix - like operating systems Main article : History of macOS The
heritage of what would become macOS had originated at NeXT , a company founded by Steve Jobs following his departure from Apple in 1985 . There , the Unix -
like NeXTSTEP operating system was developed , and then launched in 1989 . The kernel of NeXTSTEP is based upon the Mach kernel , which was originally developed
at Carnegie Mellon University , with additional kernel layers and low - level user space code derived from parts of BSD . Its graphical user interface was built
on top of an object - oriented GUI toolkit using the Objective - C programming language .
</item 9>

<item 10>
macOS
MacOS - Wikipedia MacOS Jump to : navigation , search This article is about the current Apple operating system for Mac computers . For pre-2001 versions , see
Classic Mac OS . `` OSX '' and `` OS X '' redirect here . For other uses , see OSX ( disambiguation ) . macOS Screenshot of the macOS High Sierra desktop
Developer Apple , Inc . Written in C++ Objective - C Swift OS family Macintosh , Unix Working state Current Source model Closed source ( with open source
components ) Initial release March 24 , 2001 ; 17 years ago ( 2001 - 03 - 24 ) Latest release 10.13. 4 ( 17E202 ) ( April 24 , 2018 ; 2 days ago ( 2018 - 04 -
24 ) ) ( ± ) Latest preview 10.13.
</item 10>
=== END ITEM id=0c13b76637b8aa0f ===

=== ITEM id=bc7eeef272898b0c ===
Gold sentences and facts:
Sentence 0: Year Film Role Notes Dumb and Dumber State Trooper Down Periscope E.T. `` Sonar '' Lovacelli RocketMan Fred Z . Randall Wag the Dog Pet Wrangler Space Ghost Coast to Coast Himself 1 episode 1998 Half Baked Kenny Davis There 's Something About Mary Hitchhiker ( uncredited ) Dog Park Callum Ned 's Newt Newton ( voice ) Mr. Headmistress Tucker 1999 Superstar Slater 2000 The Whole Nine Yards Special Agent Steve Hanson Big Money Hustlas Officer Harry Cox Becoming Dick Richard Breggs Freddy Got Fingered Darren Gary & Mike Mike Bonner ( voice ) 2002 Back by Midnight Sheriff Hubbard Sorority Boys Doofer / Roberta 2003 Family Tree Jake Kart Racer Zee 2005 Lucky 13 Bleckman Because of Winn - Dixie Policeman Robots Lugnut ( voice ) 2006 Employee of the Month Russell Porpis - Gunders Surf School Rip Ergo Proxy Head Dog Cop ( voice ) 1 episode 2007 Meet the Robinsons Carl ( voice ) The Dukes of Hazzard : The Beginning Sheriff Rosco P. Coltrane 2008 Bachelor Party 2 : The Last Temptation Derek Madagascar : Escape 2 Africa Giraffe ( voice ) 2009 Fudgy Wudgy Fudge Face Elmore P. Fudge My Life in Ruins Big Al Sawchuck 2009 The Haunted World of El Superbeasto Gerard the Exterminator ( voice ) 2011 Spooky Buddies Warwick the Warlock Lloyd the Conqueror Vulcan Dahmer Vs. Gacy God Dan Vs . Hiram Episode : Technology Robot Chicken Professor Cornelius Q. Quibblefingers ( voice ) 1 episode ; also writes 4 episodes 2012 The High Fructose Adventures of Annoying Orange Apple ( voice ) Robot and Monster Monster ( voice ) Main Role 2014 Back in the Day Skunk 2015 Jake and the Never Land Pirates Captain Frost ( voice ) Penn Zero : Part - Time Hero Helper Hue ( voice ) 2016 Sausage Party Ketchup Bottle / Drug Dealer / Baba Ganoush ( voice ) Skylanders Academy Hugo ( voice ) 2017 - present Puppy Dog Pals Bob ( voice ) Also show creator
  Fact 0.0: The listed performer had the role State Trooper in Dumb and Dumber.
  Fact 0.1: The listed performer appeared in Down Periscope, with no role given.
  Fact 0.2: The listed performer had the role 'Sonar' Lovacelli in E.T.
  Fact 0.3: The listed performer had the role Fred Z. Randall in RocketMan.
  Fact 0.4: The listed performer had the role Pet Wrangler in Wag the Dog.
  Fact 0.5: The listed performer had the role Himself in Space Ghost Coast to Coast, for 1 episode.
  Fact 0.6: In 1998, the listed performer had the role Kenny Davis in Half Baked.
  Fact 0.7: In 1998, the listed performer had the role Hitchhiker, uncredited, in There's Something About Mary.
  Fact 0.8: In 1998, the listed performer had the role Callum in Dog Park.
  Fact 0.9: In 1998, the listed performer had the role Newton, a voice role, in Ned's Newt.
  Fact 0.10: In 1998, the listed performer had the role Tucker in Mr. Headmistress.
  Fact 0.11: In 1999, the listed performer had the role Slater in Superstar.
  Fact 0.12: In 2000, the listed performer had the role Special Agent Steve Hanson in The Whole Nine Yards.
  Fact 0.13: In 2000, the listed performer had the role Officer Harry Cox in Big Money Hustlas.
  Fact 0.14: In 2000, the listed performer had the role Richard Breggs in Becoming Dick.
  Fact 0.15: In 2000, the listed performer had the role Darren in Freddy Got Fingered.
  Fact 0.16: In 2000, the listed performer had the role Mike Bonner, a voice role, in Gary & Mike.
  Fact 0.17: In 2002, the listed performer had the role Sheriff Hubbard in Back by Midnight.
  Fact 0.18: In 2002, the listed performer had the role Doofer / Roberta in Sorority Boys.
  Fact 0.19: In 2003, the listed performer had the role Jake in Family Tree.
  Fact 0.20: In 2003, the listed performer had the role Zee in Kart Racer.
  Fact 0.21: In 2005, the listed performer had the role Bleckman in Lucky 13.
  Fact 0.22: In 2005, the listed performer had the role Policeman in Because of Winn-Dixie.
  Fact 0.23: In 2005, the listed performer had the role Lugnut, a voice role, in Robots.
  Fact 0.24: In 2006, the listed performer had the role Russell Porpis-Gunders in Employee of the Month.
  Fact 0.25: In 2006, the listed performer had the role Rip in Surf School.
  Fact 0.26: In 2006, the listed performer had the role Head Dog Cop, a voice role, for 1 episode, in Ergo Proxy.
  Fact 0.27: In 2007, the listed performer had the role Carl, a voice role, in Meet the Robinsons.
  Fact 0.28: In 2007, the listed performer had the role Sheriff Rosco P. Coltrane in The Dukes of Hazzard: The Beginning.
  Fact 0.29: In 2008, the listed performer had the role Derek in Bachelor Party 2: The Last Temptation.
  Fact 0.30: In 2008, the listed performer had the role Giraffe, a voice role, in Madagascar: Escape 2 Africa.
  Fact 0.31: In 2009, the listed performer had the role Fudge Face Elmore P. Fudge in Fudgy Wudgy.
  Fact 0.32: In 2009, the listed performer had the role Big Al Sawchuck in My Life in Ruins.
  Fact 0.33: In 2009, the listed performer had the role Gerard the Exterminator, a voice role, in The Haunted World of El Superbeasto.
  Fact 0.34: In 2011, the listed performer had the role Warwick the Warlock, Lloyd the Conqueror, and Vulcan in Spooky Buddies.
  Fact 0.35: In 2011, the listed performer had the role God in Dahmer Vs. Gacy.
  Fact 0.36: In 2011, the listed performer had the role Robot in Dan Vs. Hiram, Episode: Technology.
  Fact 0.37: In 2011, the listed performer had the role Professor Cornelius Q. Quibblefingers, a voice role, for 1 episode, in Robot Chicken.
  Fact 0.38: The listed performer also wrote 4 episodes of Robot Chicken.
  Fact 0.39: In 2012, the listed performer had the role Apple, a voice role, in The High Fructose Adventures of Annoying Orange.
  Fact 0.40: In 2012, the listed performer had the role Monster, a voice role and a main role, in Robot and Monster.
  Fact 0.41: In 2014, the listed performer had the role Skunk in Back in the Day.
  Fact 0.42: In 2015, the listed performer had the role Captain Frost, a voice role, in Jake and the Never Land Pirates.
  Fact 0.43: In 2015, the listed performer had the role Helper Hue, a voice role, in Penn Zero: Part-Time Hero.
  Fact 0.44: In 2016, the listed performer had the role Ketchup Bottle / Drug Dealer / Baba Ganoush, a voice role, in Sausage Party.
  Fact 0.45: In 2016, the listed performer had the role Hugo, a voice role, in Skylanders Academy.
  Fact 0.46: From 2017 to the present, the listed performer had the role Bob, a voice role, in Puppy Dog Pals.
  Fact 0.47: The listed performer is also the creator of Puppy Dog Pals.

Retrieved items:
<item 1>
Harland Williams
Williams created the Disney Junior series Puppy Dog Pals that debuted in 2017 . Filmography ( edit ) Year Film Role Notes Dumb and Dumber State Trooper Down
Periscope E.T. `` Sonar '' Lovacelli RocketMan Fred Z . Randall Wag the Dog Pet Wrangler Space Ghost Coast to Coast Himself 1 episode 1998 Half Baked Kenny
Davis There 's Something About Mary Hitchhiker ( uncredited ) Dog Park Callum Ned 's Newt Newton ( voice ) Mr.
</item 1>

<item 2>
Graph query row:
person: Harland Williams
work: Dumb and Dumber
relationship: RELATED_TO
</item 2>

<item 3>
Geoffrey Palmer (actor)
( 1973 ) as Examinator Doctor / Basil Keyes The Battle of Billy 's Pond ( 1976 ) - First Policeman The Outsider ( 1979 ) as Colonel Wyndham The Honorary Consul
( 1983 ) as Belfrage : British Ambassador A Zed & Two Noughts ( 1985 ) as Fallast Clockwise ( 1986 ) as Headmaster A Fish Called Wanda ( 1988 ) as Judge Hawks (
1988 ) as SAAB Salesman The Madness of King George ( 1994 ) as Warren Mrs. Brown ( 1997 ) as Henry Ponsonby Tomorrow Never Dies ( 1997 ) as Admiral Roebuck
Stiff Upper Lips ( 1998 ) as His Butler 's Voice Anna and the King ( 1999 ) as Lord John Bradley Rat ( 2000 ) as The Doctor Peter Pan ( 2003 ) as Sir Edward
Quiller Couch Picadilly Jim ( 2005 ) - Bayliss The Pink Panther 2 ( 2009 ) as Joubert W.E. ( 2011 ) as Stanley Baldwin Lost Christmas ( 2011 ) as Dr.
</item 3>

<item 4>
Harland Williams
Headmistress Tucker 1999 Superstar Slater 2000 The Whole Nine Yards Special Agent Steve Hanson Big Money Hustlas Officer Harry Cox Becoming Dick Richard Breggs
Freddy Got Fingered Darren Gary & Mike Mike Bonner ( voice ) 2002 Back by Midnight Sheriff Hubbard Sorority Boys Doofer / Roberta 2003 Family Tree Jake Kart
Racer Zee 2005 Lucky 13 Bleckman Because of Winn - Dixie Policeman Robots Lugnut ( voice ) 2006 Employee of the Month Russell Porpis - Gunders Surf School Rip
Ergo Proxy Head Dog Cop ( voice ) 1 episode 2007 Meet the Robinsons Carl ( voice ) The Dukes of Hazzard : The Beginning Sheriff Rosco P. Coltrane 2008 Bachelor
Party 2 : The Last Temptation Derek Madagascar : Escape 2 Africa Giraffe ( voice ) 2009 Fudgy Wudgy Fudge Face Elmore P.
</item 4>

<item 5>
The Dukes of Hazzard (film)
Gainey as Sheriff Rosco P. Coltrane Michael Weston as Deputy Enos Strate Lynda Carter as Pauline James Roday as Billy Prickett Kevin Heffernan as Derek `` Sheev
'' Sheevington Nikki Griffin as Katie - Lynn Johnson Jacqui Maxwell as Annette Alice Greczyn as Laurie Pullman Junior Brown as The Balladeer ( narrator ) Joe
Don Baker as Governor Jim Applewhite Barry Corbin as Bill Pullman Andrew Prine as Angry Man Randy Orton as Bo Duke ( Stunt double ) Cameos ( edit ) All five
members of the comedy film troupe Broken Lizard make appearances in the film , classified as cameos , except for Kevin Heffernan , who had a larger speaking
role ( Sheev ) . Broken Lizard cameos Steve Lemme appears as Jimmy Pullman , the son of Bill Pullman , in an early car chase scene in which he accidentally
shoots the inside of his father 's truck . Jay Chandrasekhar and Erik Stolhanske reprise their roles as Ramathorn and Rabbit from the Broken Lizard comedy ,
Super Troopers .
</item 5>

<item 6>
The Dukes of Hazzard (film)
The characters are now campus security officers , who give a warning to the Duke boys for driving too slowly . Paul Soter appears as TV newsman Rick Shakely ,
who is reporting from the Hazzard Road Rally . Charlie Finn , who played a dimwitted fast - food employee in Super Troopers , appears as a dimwitted geology
student who assists the Duke boys with the coal samples . Other notable cameos During the bar - fight scene , Indy - car driver A.J. Foyt IV appears as himself
. Bloopers roll under the ending credits , one of which features Rip Taylor interrupting the bedroom scene with Luke and the two college girls . Taylor had
previously appeared with Knoxville in Jackass : The Movie . Production ( edit ) Stunts ( edit ) Knoxville said he was initially reluctant to take on the role ,
but was persuaded by script changes and the presence of Dan Bradley as stunt coordinator and second unit for the car chase scenes . Knoxville praised him ,
saying , `` everyone in Hollywood wants Dan Bradley to shoot their car stuff '' . Locations ( edit ) The majority of the film was shot in and around Clinton ,
Louisiana .
</item 6>

<item 7>
California Highway Patrol
Peyer declined to provide a sample , nor explain why he declined to do so . Media references ( edit ) CHiPs was a fictional television drama show ( 1977 - 1983
) about the CHP , and the CHP helped out in the 1955 TV show Highway Patrol starring Broderick Crawford . Also the CHiPs TV series was made into a modern
television film called CHiPs ' 99 . A comical motion picture version of the show has been released in 2017 ( CHiPS ) but it has not been endorsed by the CHP at
all . Former CHP Commissioner Maury Hannigan was the host of the syndicated TV series Real Stories of the Highway Patrol , which aired in the mid-to - late
1990s . The show featured not just the CHP , but also the Minnesota State Patrol , the Nevada Highway Patrol , the Alaska State Troopers , the South Dakota
Highway Patrol , the Rhode Island State Police , the Massachusetts State Police , the West Virginia State Police and many other state police and highway patrol
agencies in the United State of America , and in one episode , the Royal Newfoundland Constabulary .
</item 7>

<item 8>
The Dukes of Hazzard (film)
The cousins ' primary mode of transportation is an orange 1969 Dodge Charger that the boys affectionately refer to as the `` General Lee '' . Along the way ,
the family is tormented by corrupt Hazzard County Commissioner Jefferson Davis Hogg , widely known as `` Boss Hogg '' ( Burt Reynolds ) , and his willing but
dimwitted henchman , Sheriff Rosco P. Coltrane ( M.C. Gainey ) . After Rosco has the General Lee impounded after Bo and Luke 's attempt to run away from a
daughter of one their many moonshine customers , Billy Prickett ( James Roday ) , a famous stock - car driver , enters Hazzard to participate in the rally .
Meanwhile , Rosco plants a fake moonshine still ( `` ' cause he 's too dumb to find the real one '' ) in Uncle Jesse 's barn and seizes the Duke property in the
interest of eminent domain for Boss Hogg , forcing the family to temporarily reside with neighbor and Uncle Jesse 's love - interest , Pauline ( Lynda Carter )
.
</item 8>

<item 9>
Robbie Coltrane
Sgt . Troy 1985 Defence of the Realm Leo McAskey 1986 Caravaggio Scipione 1986 Mona Lisa Thomas Eat the Rich Jeremy 1988 Fruit Machine , The The Fruit Machine
Annabelle 1989 Henry V Falstaff 1989 Bert Rigby , You 're a Fool Sid Trample 1989 Let It Ride Ticket Seller 1989 Danny , the Champion of the World Victor Hazell
1989 Slipstream Montclaire 1990 Midnight Breaks Fat 1990 Nuns on the Run Charlie McManus / Sister Inviolata 1990 Perfectly Normal Alonzo Turner 1991 Pope Must
Die , The The Pope Must Die The Pope 1991 Triple Bogey on a Par Five Hole Steffano Baccardi Oh , What a Night Todd Nominated -- Genie Award for Best Performance
by an Actor in a Supporting Role 1993 Boswell & Johnson 's Tour of the Western Isles Dr. Samuel Johnson 1993 Adventures of Huck Finn , The The Adventures of
Huck Finn Duke 1995 GoldenEye Valentin Dmitrovich Zukovsky 1997 Buddy Dr.
</item 9>

<item 10>
The Dukes of Hazzard (film)
They are later arrested by Atlanta Police after running from campus police . Back in Hazzard , Daisy learns , with the help of Sheriff 's Deputy Enos Strate (
Michael Weston ) , that Billy Prickett has been hired by Boss Hogg to participate in the rally as a ringer . Boss Hogg then heads to Atlanta , where he informs
the Duke boys , in lock - up , that they are too late to stop him and reveals that the vote on Hogg 's proposition is at the same time as the rally , explaining
Billy Prickett 's involvement . During a transfer from detainment , Daisy helps the boys escape from the patrol car , and they speed home to try to inform the
townsfolk , escaping Atlanta Police , and the Georgia State Patrol after Bo outmaneuvers the city cops . Upon returning home , the Dukes discover that Boss Hogg
and Rosco had taken Uncle Jesse and Pauline hostage , an obvious trap for the boys , and that Billy is in on the scheme because he is ashamed of the town 's low
status .
</item 10>
=== END ITEM id=bc7eeef272898b0c ===

=== ITEM id=d1d1fad96ef75a0c ===
Gold sentences and facts:
Sentence 0: A primary ossification center is the first area of a bone to start ossifying . It usually appears during prenatal development in the central part of each developing bone . In long bones the primary centers occur in the diaphysis / shaft and in irregular bones the primary centers occur usually in the body of the bone . Most bones have only one primary center ( e.g. all long bones ) but some irregular bones such as the os coxa ( hip ) and vertebrae have multiple primary centers .
  Fact 0.0: A primary ossification center is the first area of a bone to start ossifying.
  Fact 0.1: A primary ossification center usually appears during prenatal development.
  Fact 0.2: A primary ossification center usually appears in the central part of each developing bone.
  Fact 0.3: In long bones, the primary centers occur in the diaphysis (shaft).
  Fact 0.4: In irregular bones, the primary centers usually occur in the body of the bone.
  Fact 0.5: Most bones have only one primary center, such as all long bones.
  Fact 0.6: Some irregular bones have multiple primary centers.
  Fact 0.7: The os coxa (hip) is an irregular bone with multiple primary centers.
  Fact 0.8: Vertebrae have multiple primary centers.

Retrieved items:
<item 1>
Ossification center
Anatomy , descriptive and applied . the University of California : Lea & Febiger . Development of bone Ossification Triradiate cartilage Limb development Limb
bud Apical ectodermal ridge Zone of polarizing activity Sclerotome Myotome Septum transversum Retrieved from ``
https://en.wikipedia.org/w/index.php?title=Ossification_center&oldid=808049222 '' Categories : Wikipedia articles incorporating text from the 20th edition of
Gray 's Anatomy ( 1918 ) Skeletal system Tissues ( biology ) Hidden categories : Articles needing additional references from May 2011 All articles needing
additional references Medicine infobox template using GraySubject or GrayPage Medicine infobox template using Dorlands parameter Talk About Wikipedia Nederlands
Edit links This page was last edited on 31 October 2017 , at 15 : 42 . About Wikipedia
</item 1>

<item 2>
Gallbladder
Early in development , the human embryo has three germ layers and abuts an embryonic yolk sac . During the second week of embryogenesis , as the embryo grows ,
it begins to surround and envelop portions of this sac . The enveloped portions form the basis for the adult gastrointestinal tract . Sections of this foregut
begin to differentiate into the organs of the gastrointestinal tract , such as the esophagus , stomach , and intestines . During the fourth week of
embryological development , the stomach rotates . The stomach , originally lying in the midline of the embryo , rotates so that its body is on the left . This
rotation also affects the part of the gastrointestinal tube immediately below the stomach , which will go on to become the duodenum . By the end of the fourth
week , the developing duodenum begins to spout a small outpouching on its right side , the hepatic diverticulum , which will go on to become the biliary tree .
Just below this is a second outpouching , known as the cystic diverticulum , that will eventually develop into the gallbladder . Function ( edit ) 1 .
</item 2>

<item 3>
Graph query row:
entity: primary ossification center
relationship: LOCATED_IN
relationship_detail: None
relation: None
fact: None
context: None
related: central part of a developing bone
</item 3>

<item 4>
Graph query row:
entity: primary ossification center
relationship: RELATED_TO
relationship_detail: None
relation: None
fact: None
context: None
related: Prenatal development
</item 4>

<item 5>
Brain
Once neurons have positioned themselves , their axons sprout and navigate through the brain , branching and extending as they go , until the tips reach their
targets and form synaptic connections . In a number of parts of the nervous system , neurons and synapses are produced in excessive numbers during the early
stages , and then the unneeded ones are pruned away . For vertebrates , the early stages of neural development are similar across all species . As the embryo
transforms from a round blob of cells into a wormlike structure , a narrow strip of ectoderm running along the midline of the back is induced to become the
neural plate , the precursor of the nervous system . The neural plate folds inward to form the neural groove , and then the lips that line the groove merge to
enclose the neural tube , a hollow cord of cells with a fluid - filled ventricle at the center . At the front end , the ventricles and cord swell to form three
vesicles that are the precursors of the forebrain , midbrain , and hindbrain .
</item 5>

<item 6>
Middle cranial fossa
p. 154 . ISBN 9781437735802 . Additional images ( edit ) Middle cranial fossa at human foetus Base of skull Middle cranial fossa Middle cranial fossa Middle
cranial fossa External links ( edit ) Wikimedia Commons has media related to Middle cranial fossa . Anatomy photo : 22 : os - 0802 at the SUNY Downstate Medical
Center Compound structures of skull Neurocranium Calvaria Diploë Asterion Pterion Stephanion Inion Bregma Lambda Fossae anterior middle posterior cranial cavity
Base of skull Fontanelle anterior posterior sphenoidal mastoid Facial skeleton Nasion Gonion Both dacryon zygomatic arch temporal fossa infratemporal fossa
pterygomaxillary fissure pterygopalatine fossa Foramina of the skull ( and canals , fissures , meatus , and hiatus ) Anterior cranial fossa to Orbit : ethmoidal
foramina anterior posterior to Nasal cavity : olfactory foramina ( CN - I ) foramen cecum Middle cranial fossa to Orbit : optic canal ( CN - II ) superior
orbital fissure ( CN - III ,
</item 6>

<item 7>
Ossification center
Here and there the matrix between two cells of the same row also becomes calcified , and transverse bars of calcified substance stretch across from one
calcareous column to another . Thus there are longitudinal groups of the cartilage cells enclosed in oblong cavities , the walls of which are formed of
calcified matrix which cuts off all nutrition from the cells ; the cells , in consequence , atrophy , leaving spaces called the primary areolæ . Types of
ossification centers ( edit ) Section of fetal bone of cat . ir . Irruption of the subperiosteal tissue . p . Fibrous layer of the periosteum . o . Layer of
osteoblasts . im . Subperiosteal bony deposit . There are two types of ossification centers -- primary and secondary . A primary ossification center is the
first area of a bone to start ossifying . It usually appears during prenatal development in the central part of each developing bone . In long bones the primary
centers occur in the diaphysis / shaft and in irregular bones the primary centers occur usually in the body of the bone .
</item 7>

<item 8>
Ossification center
Most bones have only one primary center ( e.g. all long bones ) but some irregular bones such as the os coxa ( hip ) and vertebrae have multiple primary centers
. A secondary ossification center is the area of ossification that appears after the primary ossification center has already appeared -- most of which appear
during the postnatal and adolescent years . Most bones have more than one secondary ossification center . In long bones , the secondary centres appear in the
epiphyses . References ( edit ) This article incorporates text in the public domain from the 20th edition of Gray 's Anatomy ( 1918 ) Jump up ^ Gray and Spitzka
( 1910 ) , page 44 . Bibliography ( edit ) Gray , Henry ; Spitzka , Edward Anthony ( 1910 ) . Anatomy , descriptive and applied . the University of California :
Lea & Febiger .
</item 8>

<item 9>
Retina
The role of Pax6 in eye development was elegantly demonstrated by Walter Gehring and colleagues , who showed that ectopic expression of Pax6 can lead to eye
formation on Drosophila antennae , wings , and legs . The optic vesicle gives rise to three structures : the neural retina , the retinal pigmented epithelium ,
and the optic stalk . The neural retina contains the retinal progenitor cells ( RPCs ) that give rise to the seven cell types of the retina . Differentiation
begins with the retinal ganglion cells and concludes with production of the Muller glia . Although each cell type differentiates from the RPCs in a sequential
order , there is considerable overlap in the timing of when individual cell types differentiate . The cues that determine a RPC daughter cell fate are coded by
multiple transcription factor families including the bHLH and homeodomain factors . In addition to guiding cell fate determination , cues exist in the retina to
determine the dorsal - ventral ( D-V ) and nasal - temporal ( N-T ) axes .
</item 9>

<item 10>
Ossification center
Ossification center - wikipedia Ossification center Jump to : navigation , search This article needs additional citations for verification . Please help improve
this article by adding citations to reliable sources . Unsourced material may be challenged and removed . ( May 2011 ) ( Learn how and when to remove this
template message ) Ossification center X-ray of ossification centers in a child . Details Identifiers Latin centrum ossificationis Dorlands / Elsevier 12227335
TA A02. 0.00. 043 FMA 75436 Anatomical terminology ( edit on Wikidata ) The first step in ossification of the cartilage is that the cartilage cells , at the
point where ossification is commencing and which is termed as an ossification center , enlarge and arrange themselves in rows . The matrix in which they are
imbedded increases in quantity , so that the cells become further separated from each other . A deposit of calcareous material now takes place in this matrix ,
between the rows of cells , so that they become separated from each other by longitudinal columns of calcified matrix , presenting a granular and opaque
appearance .
</item 10>
=== END ITEM id=d1d1fad96ef75a0c ===

=== ITEM id=79abbca2c6495eb0 ===
Gold sentences and facts:
Sentence 0: A muscle relaxant is a drug that affects skeletal muscle function and decreases the muscle tone . It may be used to alleviate symptoms such as muscle spasms , pain , and hyperreflexia . The term `` muscle relaxant '' is used to refer to two major therapeutic groups : neuromuscular blockers and spasmolytics . Neuromuscular blockers act by interfering with transmission at the neuromuscular end plate and have no central nervous system ( CNS ) activity . They are often used during surgical procedures and in intensive care and emergency medicine to cause temporary paralysis . Spasmolytics , also known as `` centrally acting '' muscle relaxants , are used to alleviate musculoskeletal pain and spasms and to reduce spasticity in a variety of neurological conditions . While both neuromuscular blockers and spasmolytics are often grouped together as muscle relaxants , the term is commonly used to refer to spasmolytics only .
  Fact 0.0: A muscle relaxant is a drug that affects skeletal muscle function.
  Fact 0.1: A muscle relaxant decreases muscle tone.
  Fact 0.2: A muscle relaxant may be used to alleviate muscle spasms.
  Fact 0.3: A muscle relaxant may be used to alleviate pain.
  Fact 0.4: A muscle relaxant may be used to alleviate hyperreflexia.
  Fact 0.5: The term 'muscle relaxant' is used to refer to two major therapeutic groups.
  Fact 0.6: The two major therapeutic groups referred to by 'muscle relaxant' are neuromuscular blockers and spasmolytics.
  Fact 0.7: Neuromuscular blockers act by interfering with transmission at the neuromuscular end plate.
  Fact 0.8: Neuromuscular blockers have no central nervous system (CNS) activity.
  Fact 0.9: Neuromuscular blockers are often used during surgical procedures.
  Fact 0.10: Neuromuscular blockers are often used in intensive care and emergency medicine.
  Fact 0.11: Neuromuscular blockers are used to cause temporary paralysis.
  Fact 0.12: Spasmolytics are also known as 'centrally acting' muscle relaxants.
  Fact 0.13: Spasmolytics are used to alleviate musculoskeletal pain and spasms.
  Fact 0.14: Spasmolytics are used to reduce spasticity in a variety of neurological conditions.
  Fact 0.15: Neuromuscular blockers and spasmolytics are often grouped together as muscle relaxants.
  Fact 0.16: The term 'muscle relaxant' is commonly used to refer to spasmolytics only.

Retrieved items:
<item 1>
Muscle relaxant
Mayo Clinic . Last Updated : April 1 , 2007 . Retrieved on September 19 , 2007 . ^ Jump up to : Miller , R.D. ( 1998 ) . `` Skeletal Muscle Relaxants '' . In
Katzung , B.G. Basic & Clinical Pharmacology ( 7th ed . ) . Appleton & Lange . pp. 434 -- 449 . ISBN 0 - 8385 - 0565 - 1 . Jump up ^ Bowman WC ( January 2006 )
. `` Neuromuscular block '' . Br. J. Pharmacol. 147 Suppl 1 : S277 -- 86 . doi : 10.1038 / sj. bjp. 0706404 . PMC 1760749 . PMID 16402115 . Jump up ^ `` Brief
History '' . Retrieved 2010 - 07 - 09 . Jump up ^ `` Brief History '' . Retrieved 2010 - 07 - 09 . Jump up ^ Craig , C.R. ; Stitzel , R.E.
</item 1>

<item 2>
Muscle relaxant
Chemical diagram of pancuronium , with red lines indicating the two acetylcholine `` molecules '' in the structure Spasmolytics ( edit ) Main article :
antispasmodic A view of the spinal cord and skeletal muscle showing the action of various muscle relaxants -- black lines ending in arrow heads represent
chemicals or actions that enhance the target of the lines , blue lines ending in squares represent chemicals or actions that inhibition the target of the line
The generation of the neuronal signals in motor neurons that cause muscle contractions are dependent on the balance of synaptic excitation and inhibition the
motor neuron receives . Spasmolytic agents generally work by either enhancing the level of inhibition , or reducing the level of excitation . Inhibition is
enhanced by mimicking or enhancing the actions of endogenous inhibitory substances , such as GABA . Terminology ( edit ) Because they may act at the level of
the cortex , brain stem or spinal cord , or all three areas , they have traditionally been referred to as `` centrally acting '' muscle relaxants . However , it
is now known not every agent in this class has CNS activity ( e.g. dantrolene ) , so this name is inaccurate .
</item 2>

<item 3>
Muscle relaxant
Other skeletal muscle relaxants of that type used around the world come from a number of drug categories and other drugs used primarily for this indication
include orphenadrine ( anticholinergic ) , chlorzoxazone , tizanidine ( clonidine relative ) , diazepam , tetrazepam and other benzodiazepines , mephenoxalone ,
methocarbamol , dantrolene , baclofen , Drugs once but no longer or very rarely used to relax skeletal muscles include meprobamate , barbiturates , methaqualone
, glutethimide and the like ; some subcategories of opioids have muscle relaxant properties , and some are marketed in combination drugs with skeletal and / or
smooth muscle relaxants such as whole opium products , some ketobemidone , piritramide and fentanyl preparations and Equagesic . Neuromuscular blockers ( edit )
Detailed view of a neuromuscular junction : 1 . Presynaptic terminal 2 . Sarcolemma 3 . Synaptic vesicle 4 . Nicotinic acetylcholine receptor 5 .
</item 3>

<item 4>
Muscle relaxant
Contents ( hide ) 1 History 2 Neuromuscular blockers 3 Spasmolytics 3.1 Terminology 3.2 Clinical use 3.3 Mechanism 4 Side effects 5 See also 6 References 7
External links History ( edit ) The earliest known use of muscle relaxant drugs was by natives of the Amazon Basin in South America who used poison - tipped
arrows that produced death by skeletal muscle paralysis . This was first documented in the 16th century , when European explorers encountered it . This poison ,
known today as curare , led to some of the earliest scientific studies in pharmacology . Its active ingredient , tubocurarine , as well as many synthetic
derivatives , played a significant role in scientific experiments to determine the function of acetylcholine in neuromuscular transmission . By 1943 ,
neuromuscular blocking drugs became established as muscle relaxants in the practice of anesthesia and surgery . The U.S. Food and Drug Administration ( FDA )
approved the use of carisoprodol in 1959 , metaxalone in August 1962 , and cyclobenzaprine in August 1977 .
</item 4>

<item 5>
Muscle relaxant
PMID 17909209 . ^ Jump up to : van Tulder MW , Touray T , Furlan AD , Solway S , Bouter LM ( 2003 ) . Van Tulder , Maurits W , ed . `` Muscle relaxants for
non-specific low back pain '' . Cochrane Database Syst Rev ( 2 ) : CD004252 . doi : 10.1002 / 14651858. CD004252 . PMID 12804507 . CS1 maint : Multiple names :
authors list ( link ) ^ Jump up to : Beebe FA , Barkin RL , Barkin S ( 2005 ) . `` A clinical and pharmacologic review of skeletal muscle relaxants for
musculoskeletal conditions '' . Am J Ther. 12 ( 2 ) : 151 -- 71 . doi : 10.1097 / 01. mjt. 0000134786.50087. d8 . PMID 15767833 . Jump up ^ See , S. ; Ginzburg
, R. ( Feb 2008 ) . `` Skeletal muscle relaxants '' .
</item 5>

<item 6>
Muscle relaxant
dantrolene ) , so this name is inaccurate . Most sources still use the term `` centrally acting muscle relaxant '' . According to MeSH , dantrolene is usually
classified as a centrally acting muscle relaxant . The World Health Organization , in its ATC , uses the term `` centrally acting agents '' , but adds a
distinct category of `` directly acting agents '' , for dantrolene . Use of this terminology dates back to at least 1973 . The term `` spasmolytic '' is also
considered a synonym for antispasmodic . Clinical use ( edit ) Spasmolytics such as carisoprodol , cyclobenzaprine , metaxalone , and methocarbamol are commonly
prescribed for low back pain or neck pain , fibromyalgia , tension headaches and myofascial pain syndrome . However , they are not recommended as first - line
agents ; in acute low back pain , they are not more effective than paracetamol or nonsteroidal anti-inflammatory drugs ( NSAIDs ) , and in fibromyalgia they are
not more effective than antidepressants .
</item 6>

<item 7>
Muscle relaxant
Retrieved 2010 - 07 - 09 . External links ( edit ) Skeletal Muscle Relaxants at the US National Library of Medicine Medical Subject Headings ( MeSH ) Skeletal
muscle relaxants ( M03 ) Peripherally acting ( primarily antinicotinic , NMJ block ) Non-depolarizing Curare alkaloids Alcuronium Dimethyltubocurarine
Tubocurarine 4 ° ammonium agents ultra-short duration : Gantacurium short duration : Mivacurium Chandonium intermediate duration : Atracurium Cisatracurium
Fazadinium Rocuronium Vecuronium long duration : Doxacurium Dimethyltubocurarine Pancuronium Pipecuronium Laudexium Gallamine unsorted : Hexafluronium (
Hexafluorenium ) Depolarizing Choline derivatives : Suxamethonium ( Succinylcholine ) Polyalkylene derivatives : Hexamethonium ACh release inhibitors Botulinum
toxin Centrally acting Carbamic acid esters Carisoprodol Cyclarbamate Difebarbamate Febarbamate Meprobamate Phenprobamate Styramate Tybamate
</item 7>

<item 8>
Muscle relaxant
Muscle relaxant - wikipedia Muscle relaxant Jump to : navigation , search This article is about skeletal muscle relaxants . For information on smooth muscle
relaxants , see Antispasmodic . A muscle relaxant is a drug that affects skeletal muscle function and decreases the muscle tone . It may be used to alleviate
symptoms such as muscle spasms , pain , and hyperreflexia . The term `` muscle relaxant '' is used to refer to two major therapeutic groups : neuromuscular
blockers and spasmolytics . Neuromuscular blockers act by interfering with transmission at the neuromuscular end plate and have no central nervous system ( CNS
) activity . They are often used during surgical procedures and in intensive care and emergency medicine to cause temporary paralysis . Spasmolytics , also
known as `` centrally acting '' muscle relaxants , are used to alleviate musculoskeletal pain and spasms and to reduce spasticity in a variety of neurological
conditions . While both neuromuscular blockers and spasmolytics are often grouped together as muscle relaxants , the term is commonly used to refer to
spasmolytics only .
</item 8>

<item 9>
Muscle relaxant
( Feb 2008 ) . `` Skeletal muscle relaxants '' . Pharmacotherapy. 28 ( 2 ) : 207 -- 13 . doi : 10.1592 / phco. 28.2. 207 . PMID 18225966 . Jump up ^ `` When Are
Muscle Relaxers Prescribed For Arthritis Patients ? '' . Retrieved 2010 - 07 - 09 . Jump up ^ MedlinePlus Encyclopedia Tension Headache Jump up ^ `` Muscle
Relaxants '' . Retrieved 2010 - 07 - 09 . Jump up ^ Rang , H.P. ; Dale , M.M. ( 1991 ) . `` Drugs Used in Treating Motor Disorders '' . Pharmacology ( 2nd ed .
) . Churchill Livingston . pp. 684 -- 705 . ISBN 044304483X . Jump up ^ Standaert , D.G. ; Young , A.B. ( 2001 ) . `` Treatment Of Central Nervous System
Degerative Disorders '' . In Goodman , L.S.
</item 9>

<item 10>
Muscle relaxant
Eur. J. Neurosci. 12 ( 2 ) : 701 -- 14 . doi : 10.1046 / j. 1460 - 9568.2000. 00955. x . PMID 10712650 . Jump up ^ Young RR , Wiegner AW ( 1987 ) . ``
Spasticity '' . Clin. Orthop . Relat . Res. 219 : 50 -- 62 . PMID 3581584 . Jump up ^ `` The Effects of Muscle Relaxant Drugs Are Powerful '' . Retrieved 2010 -
07 - 09 . Jump up ^ `` Cyclobenzaprine - Oral '' . Retrieved 2010 - 07 - 09 . Jump up ^ `` Carisoprodolx '' . Retrieved 2010 - 07 - 09 . Jump up ^ `` Side
Effects of Metaxalone -- for the Consumer '' . Retrieved 2010 - 07 - 09 . Jump up ^ `` Precautions '' . Retrieved 2010 - 07 - 09 .
</item 10>
=== END ITEM id=79abbca2c6495eb0 ===
