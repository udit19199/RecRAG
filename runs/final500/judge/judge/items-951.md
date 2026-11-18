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

=== ITEM id=fdb156f7651f42ee ===
Gold sentences and facts:
Sentence 0: Tracy McConnell , better known as `` The Mother '' , is the title character from the CBS television sitcom How I Met Your Mother . The show , narrated by Future Ted , tells the story of how Ted Mosby met The Mother . Tracy McConnell appears in 8 episodes from `` Lucky Penny '' to `` The Time Travelers '' as an unseen character ; she was first seen fully in `` Something New '' and was promoted to a main character in season 9 . The Mother is played by Cristin Milioti .
  Fact 0.0: Tracy McConnell is better known as 'The Mother'.
  Fact 0.1: Tracy McConnell is the title character from the CBS television sitcom How I Met Your Mother.
  Fact 0.2: The sitcom is narrated by Future Ted.
  Fact 0.3: The sitcom tells the story of how Ted Mosby met The Mother.
  Fact 0.4: Tracy McConnell appears in 8 episodes.
  Fact 0.5: Tracy McConnell's episodes run from 'Lucky Penny' to 'The Time Travelers'.
  Fact 0.6: Tracy McConnell appears as an unseen character in those episodes.
  Fact 0.7: Tracy McConnell was first seen fully in 'Something New'.
  Fact 0.8: Tracy McConnell was promoted to a main character in season 9.
  Fact 0.9: The Mother is played by Cristin Milioti.

Retrieved items:
<item 1>
The Mother (How I Met Your Mother)
Rolling Stone . Retrieved November 16 , 2014 . ( hide ) How I Met Your Mother Episodes Season 1 Season 2 Season 3 Season 4 Season 5 Season 6 Season 7 Season 8
Season 9 Characters Ted Mosby Marshall Eriksen Lily Aldrin Barney Stinson Robin Scherbatsky The Mother ( Tracy McConnell ) Soundtracks How I Met Your Music
Related articles Awards and nominations The Bro Code Cheerleader effect `` Nothing Suits Me Like a Suit '' Retrieved from ``
https://en.wikipedia.org/w/index.php?title=The_Mother_(How_I_Met_Your_Mother)&oldid=802354471 '' Categories : How I Met Your Mother characters Fictional
characters introduced in 2013 Fictional female musicians Hidden categories : Pages using deprecated image syntax Pages using infobox character with unknown
parameters Talk Contents About Wikipedia Español Italiano Magyar Tiếng Việt Edit links This page was last edited on 25 September 2017 , at 17 : 03 . Text is
available under the Creative Commons Attribution - ShareAlike License ; additional terms may apply .
</item 1>

<item 2>
The Mother (How I Met Your Mother)
Using Anne Hathaway and Amy Adams as examples , Thomas said that `` We did n't want it to be a big famous star because we did n't want the wider audience to
have associations with whatever actress this would be ... The whole idea is that Ted 's never seen this woman before , so it better feel that way to the
audience '' , similar to how Cobie Smulders being cast as Robin Scherbatsky had `` kept the show alive '' when it began . Bays and Thomas also did not want a
large casting call . They chose Cristin Milioti after seeing her on 30 Rock and Once ; her musical ability was also helpful , as The Mother had been described
as a band member . After competing for the role against at least two others , Milioti filmed her first scene -- for the last episode of season 8 -- having never
watched How I Met Your Mother ; she recalled , `` I had ignorance on my side . So I did n't know what it meant . '' Milioti learned of the character 's
importance only after binge watching the show during the summer .
</item 2>

<item 3>
The Mother (How I Met Your Mother)
In a flashforward in `` The Lighthouse '' , Ted proposes to the Mother at the top of the lighthouse near Farhampton Inn . She immediately accepts . In another
flashfoward in `` Unpause '' , the Mother is revealed to be pregnant with their second child , Luke , in the year 2017 . She goes into labor while she and Ted
are staying at Farhampton . Name ( edit ) The Mother 's real name is not revealed until the series finale , `` Last Forever '' . When Ted meets her at the
Farhampton train station , she reveals that her name is Tracy McConnell . In the season 1 episode `` Belly Full of Turkey '' , Ted meets a stripper named Tracy
and says `` ... that , kids , is the true story of how I met your mother '' . The children react in surprise and appear to believe Ted before he admits he is
joking , which led some fans to correctly guess that The Mother 's name is Tracy .
</item 3>

<item 4>
The Mother (How I Met Your Mother)
Jump up ^ `` Pilot '' . How I Met Your Mother . Season 1 . Episode 1 . September 19 , 2005 . CBS . ^ Jump up to : Kuchman , Bill ( April 4 , 2014 ) . `` ' How I
Met Your Mother ' ' Last Forever ' Recap : A Divisive Finale and What It Means for ' HIMYM ' '' . Retrieved July 7 , 2014 . Jump up ^ Rullo , Samantha ( 10
March 2014 ) . `` What is The Mother 's Name on ' How I Met Your Mother ' ? Episode Title Might Hold the Secret '' . Retrieved July 7 , 2014 . Jump up ^ ``
HIMYM Tomatometer Watch : Series Finale Currently Rotten - Majority of critics disappointed by ' Last Forever ' '' . Rotten Tomatoes. April 2 , 2014 . Retrieved
April 3 , 2014 .
</item 4>

<item 5>
The Mother (How I Met Your Mother)
Tracy McConnell appears in 8 episodes from `` Lucky Penny '' to `` The Time Travelers '' as an unseen character ; she was first seen fully in `` Something New
'' and was promoted to a main character in season 9 . The Mother is played by Cristin Milioti . The story of how Ted met The Mother is the framing device behind
the series ; many facts about her are revealed throughout the series , including the fact that Ted once unwittingly owned her umbrella before accidentally
leaving it behind in her apartment . Ted and The Mother meet at the Farhampton train station following Barney Stinson and Robin Scherbatsky 's wedding ; this
scene is shown in `` Last Forever '' , the series finale . The Mother 's death from an unspecified terminal illness in 2024 , also revealed in the series finale
, received a mixed reaction from fans . An alternate ending was released in the ninth season DVD . In the alternate ending , Tracy Mosby is still living when
Ted is telling the story in 2030 .
</item 5>

<item 6>
The Mother (How I Met Your Mother)
The Mother ( How I Met Your Mother ) - wikipedia The Mother ( How I Met Your Mother ) Jump to : navigation , search Tracy McConnell How I Met Your Mother
character The Mother appearing in `` The Locket '' First appearance `` Lucky Penny ( unseen ) '' `` Something New '' ( seen ) Last appearance `` Last Forever ''
Created by Carter Bays Craig Thomas Portrayed by Cristin Milioti Information Aliases The Mother Gender Female Spouse ( s ) Ted Mosby Significant other ( s ) Max
( deceased former boyfriend ) Louis ( ex-boyfriend ) Children Penny Mosby ( daughter , born in 2015 , played by Lyndsy Fonseca ) Luke Mosby ( son , born in 2017
, played by David Henrie ) Nationality American Tracy McConnell , better known as `` The Mother '' , is the title character from the CBS television sitcom How I
Met Your Mother . The show , narrated by Future Ted , tells the story of how Ted Mosby met The Mother .
</item 6>

<item 7>
The Mother (How I Met Your Mother)
In the season 8 episode `` Band or DJ ? , '' Ted runs into Cindy on the subway and tells her that the band Barney and Robin hired to play at their wedding
cancelled at the last minute . The end result of the encounter is that Cindy 's ( now ex - ) roommate 's band plays at Barney and Robin 's wedding . The Mother
is first shown meeting Louis in `` How Your Mother Met Me '' as she is left to carry the band equipment while the now - lead band member Darren talks to his
fans . Later at MacLaren 's Pub , she tells him she 's not yet ready to date . Louis asks her to give him a call if she changes her mind and they begin dating
not long after . The Mother meets all of Ted 's best friends ( Barney , Lily , Marshall and Robin ) before she meets him . The Mother is responsible for
convincing Barney to pursue Robin , as revealed through a flashback in `` Platonish '' . In `` The Locket '' , Tracy meets Lily on a train journey .
</item 7>

<item 8>
The Mother (How I Met Your Mother)
After they start talking , Mitch encourages her to pursue her dreams . The Mother expresses her desire to end poverty by taking up economics in college . On his
first day of teaching as Professor Mosby , as seen in the season 4 finale `` The Leap '' , he is seen in front of the classroom of students , one of which
Future Ted says is the titular mother . But in the first episode of season 5 , `` Definitions '' , it is revealed that he was actually in the wrong classroom --
Economics instead of Architecture . At the same time in `` How Your Mother Met Me '' , the Mother sits her first session in Economics 305 and meets another
graduate student named Cindy ( Rachel Bilson ) , whom she offers to move in with her as her roommate . They see Ted enter the room , but when he announces the
subject , The Mother thinks she is in the wrong room and runs off . She heads back to the room after seeing Ted scramble to his actual classroom . Later , in ``
Girls Versus Suits '' , Ted dates Cindy , not knowing that her roommate is his future wife .
</item 8>

<item 9>
The Mother (How I Met Your Mother)
While this incident made Cindy realise that she is a lesbian , it also made the Mother decide to go back into dating , as the kiss was her first in a long time
. Some time after this , a man named Darren approaches The Mother and is welcomed into her band named Superfreakonomics . Darren gradually takes over the band .
In the season 6 opener `` Big Days '' it is revealed Ted meets his future wife `` the day of '' the wedding at which he is the best man . In the episode ``
False Positive '' Robin asks Ted to be her future best man , should she ever get married . In the episode `` Challenge Accepted '' , it is revealed that Ted
meets the mother of his children the day of Barney 's wedding . In the last episode of season 7 , `` The Magician 's Code '' it is shown that Barney will marry
Robin , and Ted will meet the mother `` the day of '' their wedding . On the premiere of season 8 , Ted 's wife appears after Barney and Robin 's wedding ,
outside at the `` Farhampton '' station while holding a yellow umbrella and her bass guitar .
</item 9>

<item 10>
The Mother (How I Met Your Mother)
In the video , future Ted is heard saying , `` ... When I think how lucky I am to wake up next to your mom every morning , I ca n't help but be amazed how easy
it all really was ... '' , indirectly stating that The Mother is alive . The video ends right after the train passes at Farhampton station and credits start
rolling , implying that Ted never went back to Robin as he lived a successful married life with Tracy Mosby . Contents ( hide ) 1 Casting 2 Character history
2.1 Name 3 Death 4 Notes 5 References Casting ( edit ) During its first eight seasons , the successful sitcom How I Met Your Mother often hinted at the unseen
character of The Mother . Well - known actresses often made guest appearances on the show . Many fans expected that another would play one of the most - wanted
roles in Hollywood , but creators Carter Bays and Craig Thomas wanted an unknown .
</item 10>
=== END ITEM id=fdb156f7651f42ee ===

=== ITEM id=e7af04d029a72cc3 ===
Gold sentences and facts:
Sentence 0: Reading Full name Reading Football Club Nickname ( s ) The Royals Founded 1871 ; 147 years ago ( 1871 ) Ground Madejski Stadium Ground Capacity 24,161 Owner Dai Yongge and Dai Xiuli ( majority ) Chairman Sir John Madejski Manager Jaap Stam League Championship 2016 -- 17 Championship , 3rd Website Club website Home colours Away colours Current season
  Fact 0.0: The full name of the club is Reading Football Club.
  Fact 0.1: The club's nickname is The Royals.
  Fact 0.2: Reading was founded in 1871.
  Fact 0.3: Reading was founded 147 years ago.
  Fact 0.4: Reading's ground is Madejski Stadium, with a ground capacity of 24,161.
  Fact 0.5: Reading's owners are Dai Yongge and Dai Xiuli (majority).
  Fact 0.6: Reading's chairman is Sir John Madejski.
  Fact 0.7: Reading's manager is Jaap Stam.
  Fact 0.8: Reading competed in the Championship in the 2016-17 season, finishing 3rd.
  Fact 0.9: Reading's website is listed as the club website.
Sentence 1: Reading Football Club ownership structure : 75 % Owned by Renhe Sports Management Ltd , 100 % owned by Xiu Li Dai and Yongge Dai . 25 % Owned by Narin Niruttinanon
  Fact 1.0: Reading Football Club's ownership structure is 75% owned by Renhe Sports Management Ltd.
  Fact 1.1: Renhe Sports Management Ltd is 100% owned by Xiu Li Dai and Yongge Dai.
  Fact 1.2: 25% of Reading Football Club is owned by Narin Niruttinanon.

Retrieved items:
<item 1>
Reading F.C.
On 24 May 2007 , it was announced that planning permission had been granted to extend the stadium to a capacity of 36,900 . Reading have made plans for a new
training ground at Bearwood Golf Club to replace Hogwood park their current training facility . On 5 July 2016 , at the end of Eamonn Dolan 's funeral , Reading
announced that the North Stand would be renamed the `` Eamonn Dolan Stand '' . Support ( edit ) See also : Supporters ' Trust at Reading Supporters at a Reading
match at Elm Park in 1913 In 2001 , Reading became the first football club to register their fans as an official member of their squad , giving the `` player ''
registered with squad number 13 as ' Reading Fans ' . Rivalries ( edit ) Before going out of business in 1992 , Aldershot were Reading 's biggest rivals . There
was a strong rift between the two sets of fans , with fighting between fans occurring on several occasions . Strong feelings remain between fans of Reading and
fans of Aldershot Town , the refounded club in Aldershot .
</item 1>

<item 2>
Reading F.C.
. Official website Supporters ' Trust At Reading Reading F.C. on BBC Sport : Club news -- Recent results -- Upcoming fixtures Reading Football Club Current
season Under - 21s & Academy Records & Statistics Honours History History Seasons Thames Valley Royals proposal Grounds Elm Park ( 1896 -- 1998 ) Madejski
Stadium ( 1998 -- present ) People Managers Players with 1 -- 24 appearances 25 -- 99 appearances 100 + appearances Internationals Player of the Season Notable
matches Level 2 play - off finals 1995 2011 2017 Level 3 play - off finals Full Members ' Cup finals 1988 Other matches Portsmouth 7 -- 4 Reading Related
articles Hob Nob Anyone ? Supporters ' Trust At Reading Reading Women Marussia F1 Book : Reading F.C. Category : Reading F.C. Portal : Association football
Commons : Reading F.C. Reading F.C. seasons 1940 -- 41 1941 -- 42 1942 -- 43 1943 -- 44 1944 -- 45 1945 -- 46 1946 -- 47 1947 -- 48 1948 -- 49
</item 2>

<item 3>
Graph query row:
person: Dai Yongge
organization: Reading Football Club
role: majority owner
relationship: None
relation: None
fact: None
context: None
</item 3>

<item 4>
Reading F.C.
Reading were promoted to English football 's top division for the first time in their history . The 2006 -- 07 season saw Reading make their first appearance in
the top flight of English football . Reading defied pre-season predictions of relegation to finish the season in eighth place with 55 points . Reading turned
down the chance to play in the UEFA Intertoto Cup . In the run up to their second season in the Premier League , Reading took part in the 2007 Peace Cup in
South Korea . This second season was less successful , however , and Reading were relegated back to the Championship . Reading started the 2008 -- 09 season
with a 15 match unbeaten home run . They finished fourth and qualify for the play - offs , where they lost to Burnley in the semi-final . Manager Steve Coppell
resigned just hours after the game , replaced by Brendan Rodgers . Rodgers left the club by mutual consent on 16 December 2009 and Brian McDermott made
caretaker manager the same day .
</item 4>

<item 5>
Reading F.C.
internationals Former players ( edit ) Main article : List of Reading F.C. players Club officials ( edit ) Board of Directors & Senior Club Staff Chief
Executive : Ron Gourlay Commercial Director : Adam Benson Club Secretary : Sue Hewett Financial Controller : Bryan Stabler Directors : Sir John Madejski ( Co
Chairman ) , Sasima Srivikorn ( Co Chairwoman ) , Nigel Howe , Narin Niruttinanon , Sumrith Thanakarnjanasuth , Taweesuk Srisumrid and Theekharoj Piamphongsarn
Management team Director of Football : Brian Tevreden First Team Manager : Jaap Stam Assistant Manager : Andries Ulderink & Said Bakkati First Team Coach :
Michael Gilkes Goalkeeping Coach : Dave Beasant First Team Sports Scientist : Tristan Baker First Team Physio : Jon Urwin Academy Manager : Lee Herron Assistant
Academy Manager : David Dodds Academy Loan Manager : Mikele Leigertwood Kit Manager : Stewart Bannister First Team Head Scout : Steve Shorey Masseur : Dan
Buchanan Managers ( edit ) Main article : List of Reading F.C.
</item 5>

<item 6>
Reading F.C.
Reading then finished eighth in the 2006 -- 07 Premier League , their first ever season as a top flight club . Contents ( hide ) 1 History 2 Crest and colours 3
Stadium 4 Support 4.1 Rivalries 5 Sponsorship 5.1 Additional kit sponsors 6 Ownership and finances 7 Players 7.1 First - team squad 7.2 Under - 23 squad 7.3
Under - 18 squad 7.4 Out on loan 8 Records and statistics 8.1 Notable players 8.2 Player of the season 8.3 International players 8.4 Former players 9 Club
officials 10 Managers 11 Women 's team 12 Honours 12.1 Managerial 13 Affiliated clubs 14 References 15 Bibliography 16 External links History ( edit ) Main
article : History of Reading F.C. See also : Thames Valley Royals proposal Reading were formed on 25 December 1871 , following a public meeting at the Bridge
Street Rooms organised by the future club secretary Joseph Edward Sydenham .
</item 6>

<item 7>
Reading F.C.
25 % Owned by Narin Niruttinanon Reading F.C. Directors : Xiu Li Dai , Yongge Dai , Ronald Gourlay , Narin Niruttinanon , Nigel Howe Players ( edit ) As of 8
January 2018 First - team squad ( edit ) Note : Flags indicate national team as defined under FIFA eligibility rules . Players may hold more than one non-FIFA
nationality . No . Position Player GK Vito Mannone DF Chris Gunter MF Joey van den Berg 5 DF Paul McShane ( captain ) 6 DF Liam Moore 7 MF Roy Beerens 8 MF
George Evans 9 FW Joseph Mendes 10 MF John Swift 11 DF Jordan Obita 12 MF Garath McCleary 14 FW Sone Aluko 15 MF Callum Harriott 16 MF David Edwards 17 FW Modou
Barrow 18 FW Yann Kermorgant 19 MF Leandro Bacuna No .
</item 7>

<item 8>
Reading F.C.
Reading FC - Wikipedia Reading FC Jump to : navigation , search Not to be confused with Reading F.C. Women . Reading Full name Reading Football Club Nickname (
s ) The Royals Founded 1871 ; 147 years ago ( 1871 ) Ground Madejski Stadium Ground Capacity 24,161 Owner Dai Yongge and Dai Xiuli ( majority ) Chairman Sir
John Madejski Manager Jaap Stam League Championship 2016 -- 17 Championship , 3rd Website Club website Home colours Away colours Current season Reading Football
Club ( / ˈrɛdɪŋ / ( listen ) RED - ing ) is a professional association football club based in Reading , Berkshire , England . The team play in the Championship
, the second tier of English football . Reading are nicknamed The Royals , due to Reading 's location in the Royal County of Berkshire , though they were
previously known as The Biscuitmen , due to the town 's association with Huntley and Palmers .
</item 8>

<item 9>
Graph query row:
person: Dai Xiuli
organization: Reading Football Club
role: majority owner
relationship: None
relation: None
fact: None
context: None
</item 9>

<item 10>
Reading F.C.
Retrieved on 17 July 2013 . Jump up ^ HOW SOUTH GEORGIA UNITED BECAME A READING FC AFFILIATE CLUB . South Georgia United . Retrieved on 5 June 2014 .
Bibliography ( edit ) Digby , Bob ( 2001 ) . It 's a World Thing . Oxford , Oxfordshire : Oxford University Press . ISBN 978 - 0 - 19 - 913428 - 1 . Retrieved 2
June 2011 . Yeo , Stephen ( 1976 ) . Religion and Voluntary Organisations in Crisis . Abingdon , Oxfordshire : Taylor & Francis . ISBN 978 - 0 - 85664 - 017 - 9
. Retrieved 1 June 2011 . External links ( edit ) Wikimedia Commons has media related to Reading F.C. . Official website Supporters ' Trust At Reading Reading
F.C.
</item 10>
=== END ITEM id=e7af04d029a72cc3 ===

=== ITEM id=e8d75ca114b44f77 ===
Gold sentences and facts:
Sentence 0: Place Actor / Actress Episodes Finals Total score ( 175 ) 3 - 4 5 - 6 7 - 8 9 - 10 11 - 12 13 - 14 15 - 16 17 - 18 19 - 20 21 - 22 23 - 24 25 - 26 Aman Gandotra 8 12 14 14 7 + 3 = 10 9 + 3 = 12 10 + 3 = 13 10 + 4 = 14 10 + 5 = 15 10 + 5 = 15 10 + 4 = 14 Winner 141 Natasha Bharadwaj 9 18 14 15 10 + 5 = 15 10 + 5 = 15 10 + 5 = 15 10 + 5 = 15 10 + 5 = 15 10 + 5 = 15 10 + 5 = 15 Winner 166 2 / 3 Finalists Aashish Mehrotra 8 16 18 20 8 + 4 = 12 10 + 4 = 14 8 + 4 = 12 10 + 4 = 14 10 + 5 = 15 8 + 5 = 13 10 + 4 = 14 Runner up 156 Shruti Sharma 9 18 19 18 10 + 5 = 15 10 + 4 = 14 10 + 4 = 14 10 + 4 = 14 10 + 5 = 15 10 + 5 = 15 10 + 5 = 15 Runner up 166 Harshvardhan Deo NOT IN COMPETITION Debut 8 + 4 = 12 10 + 4 = 14 10 + 4 = 14 10 + 4 = 14 Runner up 54 Naina Singh 7 15 15 18 10 + 4 = 14 7 + 3 = 10 8 + 4 = 12 10 + 4 = 14 8 + 5 = 13 10 + 5 = 15 10 + 4 = 14 Runner up 147 Semifinalists Ansh Bagri 10 19 18 17 10 + 4 = 14 7 + 4 = 11 8 + 4 = 12 8 + 4 = 12 10 + 5 = 15 10 + 5 = 15 8 + 4 = 12 ELIMINATED 155 Lekha Prajapati 7 16 16 16 8 + 4 = 12 8 + 4 = 12 8 + 4 = 12 9 + 4 = 13 10 + 4 = 14 10 + 5 = 15 10 + 4 = 14 ELIMINATED 147 5 Shariq Nanda 9 16 14 14 10 + 3 = 13 8 + 4 = 12 8 + 3 = 11 10 + 4 = 14 8 + 5 = 13 ELIMINATED 116 Elisha Mayor NOT IN COMPETITION Debut 8 + 4 = 12 8 + 5 = 13 ELIMINATED 25 6 Tapan Singh NOT IN COMPETITION Debut 6 + 3 = 9 ELIMINATED 9 Simran Choudhary NOT IN COMPETITION Debut 8 + 4 = 12 ELIMINATED 12 7 Karan Taluja 7 17 16 16 6 + 3 = 9 8 + 4 = 12 6 + 3 = 9 ELIMINATED 86 Angela Krislinzki 9 18 20 16 8 + 5 = 13 6 + 3 = 9 6 + 3 = 9 ELIMINATED 97 8 Ahmed Masih 8 14 16 18 6 + 3 = 9 6 + 4 = 10 ELIMINATED 75 Kanikka Kapur 10 19 19 19 6 + 3 = 9 6 + 3 = 9 ELIMINATED 97 9 Harshvardhan Ahlawat 7 14 12 13 6 + 3 = 9 ELIMINATED 55 Pranati Rai Prakash 7 13 17 14 6 + 3 = 9 ELIMINATED 60 10 Jash Bahl 10 14 12 ELIMINATED 36 Heli Vyas 6 12 12 ELIMINATED 30 11 Shivank Chaudhary 6 12 ELIMINATED 18 Sahiba Bhasin 7 12 ELIMINATED 19 12 Karanraj Sharma 6 ELIMINATED 6 Laveena Keswani 6 ELIMINATED 6
  Fact 0.0: The table is headed Place, Actor / Actress, Episodes, Finals, and Total score (175), with round columns labeled 3-4 through 25-26.
  Fact 0.1: Aman Gandotra's total score is 141 and he is a Winner.
  Fact 0.2: Natasha Bharadwaj's total score is 166 and she is a Winner.
  Fact 0.3: Aashish Mehrotra is a Finalist with a Runner up placement and a total score of 156.
  Fact 0.4: Shruti Sharma is a Finalist with a Runner up placement and a total score of 166.
  Fact 0.5: Harshvardhan Deo was not in the competition and made a Debut, with a Runner up placement and a total score of 54.
  Fact 0.6: Naina Singh is a Finalist with a Runner up placement and a total score of 147.
  Fact 0.7: Ansh Bagri is a Semifinalist who was ELIMINATED with a total score of 155.
  Fact 0.8: Lekha Prajapati is a Semifinalist who was ELIMINATED with a total score of 147.
  Fact 0.9: Shariq Nanda was ELIMINATED with a total score of 116.
  Fact 0.10: Elisha Mayor was not in the competition, made a Debut, was ELIMINATED, and has a total score of 25.
  Fact 0.11: Tapan Singh was not in the competition, made a Debut, was ELIMINATED, and has a total score of 9.
  Fact 0.12: Simran Choudhary was not in the competition, made a Debut, was ELIMINATED, and has a total score of 12.
  Fact 0.13: Karan Taluja was ELIMINATED with a total score of 86.
  Fact 0.14: Angela Krislinzki was ELIMINATED with a total score of 97.
  Fact 0.15: Ahmed Masih was ELIMINATED with a total score of 75.
  Fact 0.16: Kanikka Kapur was ELIMINATED with a total score of 97.
  Fact 0.17: Harshvardhan Ahlawat was ELIMINATED with a total score of 55.
  Fact 0.18: Pranati Rai Prakash was ELIMINATED with a total score of 60.
  Fact 0.19: Jash Bahl was ELIMINATED with a total score of 36.
  Fact 0.20: Heli Vyas was ELIMINATED with a total score of 30.
  Fact 0.21: Shivank Chaudhary was ELIMINATED with a total score of 18.
  Fact 0.22: Sahiba Bhasin was ELIMINATED with a total score of 19.
  Fact 0.23: Karanraj Sharma was ELIMINATED with a total score of 6.
  Fact 0.24: Laveena Keswani was ELIMINATED with a total score of 6.
Sentence 1: India 's Next Superstars is a talent - searcher Indian reality show , which premiered on Star Plus and is streamed on Hotstar . Karan Johar and Rohit Shetty are the judges for the show . Aman Gandotra and Natasha Bharadwaj are declared winners of 2018 season , alongside Shruti Sharma who won a ' Special Mention ' award . Runners up in the male category were Aashish Mehrotra and Harshvardhan Deo and in the female category were Naina Singh and Shruti Sharma .
  Fact 1.0: India's Next Superstars is an Indian talent-searcher reality show.
  Fact 1.1: India's Next Superstars premiered on Star Plus.
  Fact 1.2: India's Next Superstars is streamed on Hotstar.
  Fact 1.3: Karan Johar and Rohit Shetty are the judges for India's Next Superstars.
  Fact 1.4: Aman Gandotra and Natasha Bharadwaj are declared winners of the 2018 season.
  Fact 1.5: Shruti Sharma is declared a winner of the 2018 season alongside Aman Gandotra and Natasha Bharadwaj.
  Fact 1.6: Shruti Sharma won a 'Special Mention' award.
  Fact 1.7: Aashish Mehrotra and Harshvardhan Deo were the runners up in the male category.
  Fact 1.8: Naina Singh and Shruti Sharma were the runners up in the female category.

Retrieved items:
<item 1>
India's Next Superstars
ELIMINATED ON 18 MARCH 2018 Heli Vyas ELIMINATED ON 4 FEBRUARY 2018 Kanikka Kapur ELIMINATED ON 25 FEBRUARY 2018 Laveena Keswani ELIMINATED ON 21 JANUARY 2018
Lekha Prajapati SEMIFINALIST ; ELIMINATED ON 1 APRIL 2018 Naina Singh RUNNER UP Natasha Bharadwaj WINNER Pranati Rai Prakash ELIMINATED ON 18 FEBRUARY 2018
Sahiba Bhasin ELIMINATED ON 28 JANUARY 2018 Simran Choudhary WILD - CARD ENTRY ON 3 MARCH 2018 ; ELIMINATED ON 11 MARCH 2018 Shruti Sharma 2nd Winner Guest
appearance ( edit ) Episode 1 & 2 - Priyanka Chopra Episode 3 & 4 - Sidharth Malhotra Episode 5 & 6 - Kangana Ranaut Episode 9 & 10 - Kartik Aaryan ,
</item 1>

<item 2>
India's Next Superstars
Quizzing in India Sell Me the Answer Thatt Antha Heli Singing Airtel Super Singer Airtel Super Singer Junior Fame Gurukul Idea Star Singer Indian Idol Sa Re Ga
Ma Pa ( Sa Re Ga Ma Pa L'il Champs ) Voice of India Lip Sing Battle Sports Box Cricket League BCL Punjab Zee 20 Cricket League Others Bigg Boss Bigg Boss Bangla
Bigg Boss Tamil Bigg Boss Kannada Bigg Boss Telugu Biggest Loser Jeetega India 's Got Talent Lead India Malayalee House National Bingo Night Sacch Ka Saamna
Defunct Ad Mad Show Antakshari -- The Great Challenge Baazi Kiski Biggest Loser Jeetega Boogie Woogie Champions of the World Dance Premier League Deal Ya No
Deal Fame X Idea Star Singer India 's Magic Star Iss Jungle Se Mujhe Bachao Jo Jeeta Wohi Super Star Junoon Lead India Mastermind India Perfect Bride National
Bingo Night Sawaal Dus Crore Ka Swayamwar Titan Antakshari Zee Cinestars Zor Ka Jhatka Retrieved from `` https://en.wikipedia.org/w/index.
</item 2>

<item 3>
India's Next Superstars
Contents ( hide ) 1 Concept 2 Contestants 3 Guest appearance 4 Scoring chart 5 Synopsis of Episodes 6 Lights Camera Superstar cocktails 7 Superstars Ki
Paathshaala 8 References 9 External links Concept ( edit ) The new talent hunt show of Karan Johar and Rohit Shetty is concerned , ten males and ten females
will be living under one roof , and every day they will be assigned for tasks and will be judged on the basis of their performances . Additionally , two more
males and females , will join the competition as Wild - Cards . The two winners will be rewarded with a three - film deal with Dharma Productions house .
Contestants ( edit ) BOYS Names Notes Aashish Mehrotra RUNNER UP Ahmed Masih ELIMINATED ON 25 FEBRUARY 2018 Aman Gandotra WINNER Ansh Bagri SEMIFINALIST ;
</item 3>

<item 4>
India's Next Superstars
India 's next Superstars - Wikipedia India 's next Superstars Jump to : navigation , search This article may require copy editing for MOS issues (
Capitalization , bold , etc . ) . You can assist by editing it . ( April 2018 ) ( Learn how and when to remove this template message ) India 's Next Superstars
Genre Reality Based on Acting Presented by Rithvik Dhanjani Karan Wahi Judges INS Judges : Rohit Shetty Karan Johar INS Pathshala Judges : Mukesh Chhabra Mahesh
Bhatt Country of origin India No. of seasons No.
</item 4>

<item 5>
India's Next Superstars
of seasons No. of episodes INS : 25 ( as of 7 April 2018 ) INS Pathshala : 60 ( as of 6 April 2018 ) Production Running time 1 hr 30 mins Production company ( s
) Dharma Productions Endemol Shine India Distributor Star India Release Original network STAR Plus Picture format 720p SDTV 1080p HDTV Original release 13
January ( 2018 - 01 - 13 ) -- 8 April 2018 External links Hotstar India 's Next Superstars is a talent - searcher Indian reality show , which premiered on Star
Plus and is streamed on Hotstar . Karan Johar and Rohit Shetty are the judges for the show . Aman Gandotra and Natasha Bharadwaj are declared winners of 2018
season , alongside Shruti Sharma who won a ' Special Mention ' award . Runners up in the male category were Aashish Mehrotra and Harshvardhan Deo and in the
female category were Naina Singh and Shruti Sharma .
</item 5>

<item 6>
India's Next Superstars
wikipedia.org/w/index.php?title=India%27s_Next_Superstars&oldid=837242330 '' Categories : Indian reality television series Indian game shows Hindi - language
television programs 2018 Indian television series debuts Star Plus television series Hidden categories : Wikipedia articles needing copy edit from April 2018
All articles needing copy edit Use dmy dates from November 2017 Use Indian English from November 2017 All Wikipedia articles written in Indian English Pages
using infobox television with alias parameters Talk Contents About Wikipedia বাংলা Edit links This page was last edited on 19 April 2018 , at 16 : 04 . Text is
available under the Creative Commons Attribution - ShareAlike License ; additional terms may apply . By using this site , you agree to the Terms of Use and
Privacy Policy . Wikipedia ® is a registered trademark of the Wikimedia Foundation , Inc. , a non-profit organization . About Wikipedia
</item 6>

<item 7>
India's Next Superstars
14 14 7 + 3 = 10 9 + 3 = 12 10 + 3 = 13 10 + 4 = 14 10 + 5 = 15 10 + 5 = 15 10 + 4 = 14 Winner 141 Natasha Bharadwaj 9 18 14 15 10 + 5 = 15 10 + 5 = 15 10 + 5 =
15 10 + 5 = 15 10 + 5 = 15 10 + 5 = 15 10 + 5 = 15 Winner 166 2 / 3 Finalists Aashish Mehrotra 8 16 18 20 8 + 4 = 12 10 + 4 = 14 8 + 4 = 12 10 + 4 = 14 10 + 5 =
15 8 + 5 = 13 10 + 4 = 14 Runner up 156 Shruti Sharma 9 18 19 18 10 + 5 = 15 10 + 4 = 14 10 + 4 =
</item 7>

<item 8>
India's Next Superstars
Then after some leg pulling , the next set of guests was called in - ' ' Gyancho ' ' baba Aashish and Natasha . Natasha was handcuffed and blind - folded , and
then Karan and Aashish touched her with ice cubes , chocolate sauce , brush , balloon and other stuffs . Then was the turn of Aashish . Both of them sang item
songs like ' ' Oo la la , ' ' ' ' baby doll main sone di , ' ' kaate nahin katte yeh din yeh raat ' ' to make the game seductive . Finally Aashish won the gift
hamper . Episode 10 This episode was telecast on 06 Apr 2018 ( Friday ) on hotstar . The finale episode started with KJo and Rohit Shetty speculating about the
possibilities of different winners of this show . All the technicians were sad because the show was ending and they did n't have the next project in hand .
First set of guests were Harshvardhan , Natasha and Shruti . Karan Wahi showed the superstars their past and future via a Time Machine which was created by him
.
</item 8>

<item 9>
India's Next Superstars
Retrieved 2018 - 04 - 08 . External links ( edit ) India 's Next Superstars Streaming on Hotstar Current broadcasts on Star Plus Currently broadcast Primetime
soap opera Har Shaakh Pe Ullu Baithaa Hai Ikyawann Ishqbaaaz Kullfi Kumarr Bajewala Naamkarann Tu Sooraj , Main Saanjh Piyaji Yeh Hai Mohabbatein Yeh Rishta Kya
Kehlata Hai Weekend Khichdi Reality television shows in India Adventure Fear Factor India ( Fear Factor : Khatron Ke Khiladi ) Survivor India Roadies Comedy
Comedy Circus Comedy Nights Bachao Comedy Nights Live Comedy Nights with Kapil Majaa Talkies Comedy Superstars Laugh India Laugh Lunars Comedy Express Mirakkel
The Great Indian Laughter Challenge The Kapil Sharma Show Dance D 4 Dance Dance Bangla Dance Dance India Dance Dance Plus Dance Champions Jhalak Dikhhla Jaa
Jodi Number One Just Dance Nach Baliye Dating MTV Splitsvilla Power Couple Food Dhe Chef Foodistan MasterChef India Kitchen Champion Game 10 Ka Dum Deal Ya No
Deal India Poochega Sabse Shaana Kaun ?
</item 9>

<item 10>
India's Next Superstars
Gandotra was impressed with the wigs and the earrings which were given to them along with the costumes to make their acts look perfect . References ( edit )
Jump up ^ `` Karan Johar , Rohit Shetty to judge a new talent show '' . DNA . Chaya Unnikrishnan . Retrieved November 3 , 2017 . Jump up ^ `` Masaba Gupta
designs Karan Johar 's outfits for India 's Next Superstars '' . Firstpost . IANS . Retrieved November 16 , 2017 . Jump up ^ `` Exclusive Aman Gandotra ,
Natasha Bharadwaj and Shruti Sharma were declared winners of India 's Next Superstars '' . The Indian Express . 2018 - 04 - 07 . Retrieved 2018 - 04 - 08 .
</item 10>
=== END ITEM id=e8d75ca114b44f77 ===

=== ITEM id=362d5af72e4b7d0e ===
Gold sentences and facts:
Sentence 0: The ABA Model Rules of Professional Conduct , created by the American Bar Association ( ABA ) , are a set of rules that prescribe baseline standards of legal ethics and professional responsibility for lawyers in the United States . They were promulgated by the ABA House of Delegates upon the recommendation of the Kutak Commission in 1983 . The rules are merely recommendations , or models , ( hence the name `` Model Rules '' ) and are not themselves binding . However , having a common set of Model Rules facilitates a common discourse on legal ethics , and simplifies professional responsibility training as well as the day - to - day application of such rules . As of 2015 , 49 states and four territories have adopted the rules in whole or in part , of which the most recent to do so was the Commonwealth of the Northern Mariana Islands in March 2015 . California is the only state that has not adopted the ABA Model Rules , while Puerto Rico is the only U.S. jurisdiction outside of confederation has not adopted them but instead has its own Código de Ética Profesional .
  Fact 0.0: The ABA Model Rules of Professional Conduct were created by the American Bar Association (ABA).
  Fact 0.1: The ABA Model Rules of Professional Conduct are a set of rules.
  Fact 0.2: The ABA Model Rules prescribe baseline standards of legal ethics for lawyers in the United States.
  Fact 0.3: The ABA Model Rules prescribe baseline standards of professional responsibility for lawyers in the United States.
  Fact 0.4: The ABA House of Delegates promulgated the rules upon the recommendation of the Kutak Commission.
  Fact 0.5: The Kutak Commission's recommendation came in 1983.
  Fact 0.6: The rules are merely recommendations, or models.
  Fact 0.7: The name Model Rules reflects that the rules are recommendations or models.
  Fact 0.8: The rules are not themselves binding.
  Fact 0.9: Having a common set of Model Rules facilitates a common discourse on legal ethics.
  Fact 0.10: Having a common set of Model Rules simplifies professional responsibility training.
  Fact 0.11: Having a common set of Model Rules simplifies the day-to-day application of such rules.
  Fact 0.12: As of 2015, 49 states and four territories had adopted the rules in whole or in part.
  Fact 0.13: The most recent jurisdiction to adopt the rules as of 2015 was the Commonwealth of the Northern Mariana Islands.
  Fact 0.14: The Commonwealth of the Northern Mariana Islands adopted the rules in March 2015.
  Fact 0.15: California is the only state that has not adopted the ABA Model Rules.
  Fact 0.16: Puerto Rico is the only U.S. jurisdiction outside of confederation that has not adopted the ABA Model Rules.
  Fact 0.17: Puerto Rico has its own Codigo de Etica Profesional instead.

Retrieved items:
<item 1>
American Bar Association Model Rules of Professional Conduct
In 2001 , the Commission for the Revision of the Rules of Professional Conduct of the State Bar of California began a comprehensive revision of the California
rules that would , among other things , convert them into a localized version of the ABA Model Rules . However , the Commission 's progress had been extremely
slow , simply because there are so many substantive and structural differences between the California rules and the ABA models . In October 2010 , the State Bar
of California formally adopted the Commission 's report , recommending adoption of 67 proposed changes to harmonise the California Rules of Professional Conduct
with the ABA Model Rules of Professional Responsibility . On September 19 , 2014 , however , for reasons that were not fully explained , the Supreme Court of
California rejected all of the proposed revisions and returned them to the State Bar . The Court 's letter directed the State Bar to start the process all over
again with a new Commission , and to submit a new set of revised rules by March 31 , 2017 . Professional regulation ( edit ) Except for the Patent Bar , there
is no national legal profession in the United States but there are 56 separate legal professions .
</item 1>

<item 2>
The ABA Model Rules of Professional Conduct, recommended by the Kutak Commission and adopted by the ABA House of Delegates in 1983, set baseline ethical
standards covering lawyers’ duties to clients, courts, and the profession. They are nonbinding models, but jurisdictions make adopted versions enforceable. By
2015, 49 states, D.C., and four territories had adopted them wholly or partly; California retained distinct rules, while Puerto Rico had its own code. State
variations include New
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
Even though New York did not adopt the Model Rules verbatim , the advantage of adopting its overall structure is that it simplifies the professional
responsibility training of New York lawyers , and makes it easier for out - of - state lawyers to conform their conduct to New York rules by simply comparing
their home state 's version of the Model Rules to New York 's version . In June 2009 , the Supreme Court of Maine approved the adoption of the Model Rules in
that state , effective August 1 , 2009 . Among the portions of the Model Rules that the Maine Rules of Professional Conduct do not include is Model Rule 1.8 ( j
) ( 2002 ) categorically prohibiting sexual relations between lawyer and client . In March 2015 , the Supreme Court of the Commonwealth of the Northern Mariana
Islands adopted its current Rules of Attorney Discipline and Procedure , which incorporates the ABA Model Rules by reference and expands upon them by adding
another 22 local rules of professional conduct . California ( edit ) California still has its own set of unique rules governing professional attorney conduct .
</item 4>

<item 5>
American Bar Association Model Rules of Professional Conduct
External links ( edit ) The ABA Model Rules of Professional Conduct Pre-2002 Changes to the Model Rules since its adoption in 1983 from Cornell Law School
Retrieved from `` https://en.wikipedia.org/w/index.php?title=American_Bar_Association_Model_Rules_of_Professional_Conduct&oldid=789593953 '' Categories : 1983
in law Legal ethics Codes of conduct American Bar Association United States law 1983 documents Hidden categories : Articles containing potentially dated
statements from 2015 All articles containing potentially dated statements Talk Contents About Wikipedia 한국어 Suomi Edit links This page was last edited on 8 July
2017 , at 08 : 30 . Text is available under the Creative Commons Attribution - ShareAlike License ; additional terms may apply . By using this site , you agree
to the Terms of Use and Privacy Policy . Wikipedia ® is a registered trademark of the Wikimedia Foundation , Inc. , a non-profit organization . About Wikipedia
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
American Bar Association Model Rules of Professional Conduct
This is due to California law and public policy being so markedly different from the rest of the country in so many ways . For example , California is one of
only four states where covenants not to compete are void as against public policy ; and it is one of only three states where trusts are presumed to be revocable
unless the trust instrument explicitly states the trust is irrevocable ( whereas the opposite is true in the other 47 states ) . By way of comparative example ,
California has long held the position that clients must be encouraged to speak candidly to their legal counsel about both past and future actions knowing their
statements will be held in complete confidence . Their counsel can then explain the relevant law to them and urge them to conform their future conduct to the
law , thus promoting lawful conduct . In contrast , ABA Model Rule 1.6 contains exceptions to confidentiality for both violent and nonviolent ( e.g. financial )
future crimes , which California believes creates a strong disincentive for clients to discuss their future plans with their attorneys - but for which the
client might chose to comply with the law instead of violating it .
</item 7>

<item 8>
American Bar Association Model Rules of Professional Conduct
However , the Model Rules have been adopted , in whole or in part , and sometimes in variation , as the rules of professional conduct for attorneys in 49 states
, the District of Columbia , and four of the five inhabited U.S. territories . The rules of professional conduct or professional responsibility adopted in a
particular state are legally enforceable against the lawyers of that state as well as any lawyer practicing there on a pro hac vice basis . On December 17 ,
2008 , New York announced that it would finally abandon the old ABA Model Code of Professional Responsibility ( the last state to do so ) and adopt a heavily
modified version of the ABA Model Rules , effective April 1 , 2009 . New York 's version of the Model Rules was created by adjusting the standard Model Rules to
reflect indigenous New York rules that had been incorporated over the years into its version of the Model Code .
</item 8>

<item 9>
American Bar Association Model Rules of Professional Conduct
As of 2015 , 49 states and four territories have adopted the rules in whole or in part , of which the most recent to do so was the Commonwealth of the Northern
Mariana Islands in March 2015 . California is the only state that has not adopted the ABA Model Rules , while Puerto Rico is the only U.S. jurisdiction outside
of confederation has not adopted them but instead has its own Código de Ética Profesional . Contents ( hide ) 1 State , commonwealth and territorial adoptions
of the Rules 1.1 California 2 Professional regulation 3 See also 4 References 5 External links State , Commonwealth and territorial adoptions of the Rules (
edit ) The American Bar Association is a private sector voluntary professional association . As such , it has no lawmaking power or regulatory authority , but
it is just like any other nongovernmental bar association . Accordingly , the Model Rules are not legally binding on ABA members or within any particular
jurisdiction .
</item 9>

<item 10>
American Bar Association Model Rules of Professional Conduct
California ( edit ) California still has its own set of unique rules governing professional attorney conduct . Although many of the rules are similar and
sometimes identical to the ABA Model Rules , the California rules differ markedly from the ABA models in both format , scope and content . For example , under
the ABA Model Rules , an attorney may be disciplined for the appearance of impropriety , regardless whether any actual impropriety occurred ; whereas the
California rules permit discipline for only actual impropriety . This reflects the nature of California law . It is unlikely that California will ever adopt the
ABA Model Rules apart beyond harmonising the format and existing California rules that are identical or nearly identical to ABA Model Rules . Reasons for this
reluctance include the fact there are more than 280,000 attorneys licensed in California , so a drastic change from the existing rules to completely different
rules would shock the profession and , ultimately , would most likely fail . Additionally , many of California 's rules of professional conduct are established
by statute and , thus , would require action by the California Legislature to change them ; or , alternatively , action by the people of California through the
voter initiative process .
</item 10>
=== END ITEM id=362d5af72e4b7d0e ===

=== ITEM id=f6e9922dfc1829ec ===
Gold sentences and facts:
Sentence 0: In 1991 , the brainwashed super-soldier James `` Bucky '' Barnes is dispatched from a Hydra base in Siberia to intercept an automobile carrying a case of super-soldier serum . In the present day , approximately one year after Ultron 's defeat in the nation of Sokovia at the hands of the Avengers , Steve Rogers , Natasha Romanoff , Sam Wilson , and Wanda Maximoff stop Brock Rumlow from stealing a biological weapon from a lab in Lagos . Rumlow blows himself up , hoping to kill Rogers . When Maximoff throws the explosion into the sky with telekinesis , it damages a nearby building , killing several Wakandan humanitarian workers .
  Fact 0.0: In 1991, the brainwashed super-soldier James 'Bucky' Barnes was dispatched from a Hydra base in Siberia.
  Fact 0.1: In 1991, James 'Bucky' Barnes was dispatched to intercept an automobile carrying a case of super-soldier serum.
  Fact 0.2: In the present day, Ultron's defeat in the nation of Sokovia at the hands of the Avengers happened approximately one year earlier.
  Fact 0.3: Ultron was defeated in the nation of Sokovia by the Avengers.
  Fact 0.4: In the present day, Steve Rogers, Natasha Romanoff, Sam Wilson, and Wanda Maximoff stop Brock Rumlow from stealing a biological weapon from a lab in Lagos.
  Fact 0.5: Brock Rumlow blows himself up.
  Fact 0.6: Brock Rumlow hoped to kill Steve Rogers by blowing himself up.
  Fact 0.7: Wanda Maximoff throws the explosion into the sky with telekinesis.
  Fact 0.8: The explosion damages a nearby building.
  Fact 0.9: The explosion killed several Wakandan humanitarian workers.

Retrieved items:
<item 1>
Captain America: Civil War
It is the sequel to 2011 's Captain America : The First Avenger and 2014 's Captain America : The Winter Soldier , and the thirteenth film in the Marvel
Cinematic Universe ( MCU ) . The film is directed by Anthony and Joe Russo , with a screenplay by Christopher Markus & Stephen McFeely , and features an
ensemble cast , including Chris Evans , Robert Downey Jr. , Scarlett Johansson , Sebastian Stan , Anthony Mackie , Don Cheadle , Jeremy Renner , Chadwick
Boseman , Paul Bettany , Elizabeth Olsen , Paul Rudd , Emily VanCamp , Tom Holland , Frank Grillo , William Hurt , and Daniel Brühl . In Captain America : Civil
War , disagreement over international oversight of the Avengers fractures them into opposing factions -- one led by Steve Rogers and the other by Tony Stark .
Development of Civil War began in late 2013 when Markus and McFeely began writing the screenplay , which borrows concepts from the 2006 comic book storyline ``
Civil War '' , while also focusing on story and character elements from the previous Captain America films to conclude the trilogy .
</item 1>

<item 2>
Captain America: Civil War
^ Jump up to : Brett , Jennifer ( May 11 , 2015 ) . `` Photos from the `` Captain America : Civil War '' set in Atlanta `` . Atlanta Journal - Constitution .
Archived from the original on May 12 , 2015 . Retrieved May 12 , 2015 . Jump up ^ Brett , Jennifer ( May 13 , 2015 ) . `` Photos from the `` Captain America :
Civil War '' set in Atlanta `` . Atlanta Journal - Constitution . Archived from the original on May 14 , 2015 . Retrieved May 14 , 2015 . Jump up ^ Brett ,
Jennifer ( May 15 , 2015 ) . `` Photos : Chris Evans on the Atlanta `` Captain America '' set `` . Atlanta Journal - Constitution . Archived from the original
on May 17 , 2015 . Retrieved May 15 , 2015 . ^ Jump up to : Leslie , Jennifer ( May 5 , 2016 ) . `` '' Captain America : Civil War '' was shot in , around
Atlanta `` . WXIA - TV .
</item 2>

<item 3>
Captain America: Civil War
Following the November 2014 hacking of Sony Pictures ' computers , emails between Sony Pictures Entertainment co-chairman Amy Pascal and president Doug Belgrad
were released stating that Marvel wanted to include Spider - Man ( whose film rights are licensed to Sony ) in the film , but talks between the studios
concerning this were believed to have broken down . However , in February 2015 , Sony Pictures and Marvel Studios reached a licensing deal for the use of Spider
- Man in an MCU film , and reports indicated that the character would indeed appear in Civil War . The Russos stated they were lobbying for months to include
the character in the film . In January 2015 , Mackie revealed that , in addition to Atlanta , filming locations would include Puerto Rico and Berlin , while the
Russo brothers confirmed that Scarlett Johansson would return in the film as Natasha Romanoff / Black Widow . Editor Jeffrey Ford , who worked on The Winter
Soldier , also signed on for Civil War . In March 2015 , Jeremy Renner was revealed to be reprising his role as Clint Barton / Hawkeye .
</item 3>

<item 4>
Captain America: Civil War
Contents ( hide ) 1 Plot 2 Cast 3 Production 3.1 Development 3.2 Pre-production 3.3 Filming 3.4 Post-production 3.4. 1 Visual effects 4 Music 5 Release 5.1
Marketing 5.1. 1 Merchandise 5.2 Home media 6 Reception 6.1 Box office 6.1. 1 United States and Canada 6.1. 2 Other territories 6.2 Critical response 6.3
Accolades 7 Future 8 Notes 9 References 10 External links Plot ( edit ) In 1991 , the brainwashed super-soldier James `` Bucky '' Barnes is dispatched from a
Hydra base in Siberia to intercept an automobile carrying a case of super-soldier serum . In the present day , approximately one year after Ultron 's defeat in
the nation of Sokovia at the hands of the Avengers , Steve Rogers , Natasha Romanoff , Sam Wilson , and Wanda Maximoff stop Brock Rumlow from stealing a
biological weapon from a lab in Lagos . Rumlow blows himself up , hoping to kill Rogers .
</item 4>

<item 5>
Captain America: Civil War
The next month , it was revealed that the film would be converted to 3D in post-production , and that Brühl would be playing Helmut Zemo . Additionally ,
Elizabeth Olsen revealed she would reprise her role in the film as Wanda Maximoff / Scarlet Witch . Filming ( edit ) Film set for Captain America : Civil War in
Downtown Atlanta Principal photography began on April 27 , 2015 , in Fayette County , Georgia at Pinewood Atlanta Studios , under the working title Sputnik .
Other filming locations in the Atlanta metropolitan area included the Buckhead district of Atlanta ; the Peachtree Christian Church in Midtown Atlanta ;
Downtown Atlanta , including an area near Philips Arena known as The Gulch , which served as the market in Lagos ; Norcross , Georgia ; Porsche 's headquarters
in Atlanta at Aerotropolis Atlanta , which served as the Avengers ' headquarters in the film ; and the Atlanta Civic Center , which served as the IFID (
Institute for Infectious Diseases ) laboratory in Lagos ; the latter 's interiors doubled as MIT , and a third part was used for a Berlin location .
</item 5>

<item 6>
Captain America: Civil War
Following positive reactions to test screenings of The Winter Soldier , the Russo brothers were brought back to direct in early 2014 . The film 's title and
premise were revealed in October 2014 , along with Downey 's casting ; additional cast members joined in the following months . Principal photography began in
April 2015 at Pinewood Atlanta Studios in Fayette County , Georgia , and continued in the Metro Atlanta area before concluding in Germany in August 2015 , with
the film being the first to use IMAX 's digital 2D cameras ( for the film 's central airport fight sequence ) . Visual effects were provided by nearly 20
different studios during the post-production process . Captain America : Civil War held its world premiere in Los Angeles on April 12 , 2016 , and was released
in the United States on May 6 , 2016 , in 3D and IMAX 3D . The film became a critical and commercial success , grossing over $1.1 billion worldwide , making it
the highest - grossing film of 2016 and , at the time , the 12th - highest - grossing film ever .
</item 6>

<item 7>
Captain America: Civil War
wikipedia.org/w/index.php?title=Captain_America:_Civil_War&oldid=819487324 '' Categories : 2016 films English - language films 2010s action thriller films 2010s
adventure films 2010s psychological thriller films 2010s science fiction films 2010s superhero films 2016 3D films American 3D films American action thriller
films American films American political thriller films American psychological thriller films American science fiction action films American science fiction
thriller films American sequel films Babelsberg Studio films Captain America films Crossover films Drone films Films scored by Henry Jackman Films about revenge
Films about terrorism Films based on works by Mark Millar Films directed by Anthony and Joe Russo Films set in 1991 Films set in Berlin Films set in Bucharest
Films set in Cleveland Films set in Germany Films set in Lagos Films set in London Films set in Massachusetts Films set in New York ( state ) Films set in
Queens , New York Films set in Siberia Films set in Vienna Films set in prison Films set in a fictional African country Films shot at Pinewood Atlanta Studios
Films shot in Atlanta Films shot in Berlin Films shot in Brazil Films shot in Germany Films shot in Norway Films shot
</item 7>

<item 8>
Captain America: Civil War
Captain America : Civil War - Wikipedia Captain America : Civil War Jump to : navigation , search Captain America : Civil War Theatrical release poster Directed
by Anthony Russo Joe Russo Produced by Kevin Feige Screenplay by Christopher Markus Stephen McFeely Based on Captain America by Joe Simon Jack Kirby Starring
Chris Evans Robert Downey Jr . Scarlett Johansson Sebastian Stan Anthony Mackie Don Cheadle Jeremy Renner Chadwick Boseman Paul Bettany Elizabeth Olsen Paul
Rudd Emily VanCamp Tom Holland Frank Grillo William Hurt Daniel Brühl Music by Henry Jackman Cinematography Trent Opaloch Edited by Jeffrey Ford Matthew Schmidt
Production company Marvel Studios Distributed by Walt Disney Studios Motion Pictures Release date April 12 , 2016 ( 2016 - 04 - 12 ) ( Dolby Theatre ) May 6 ,
2016 ( 2016 - 05 - 06 ) ( United States ) Running time 147 minutes Country United States Language English Budget $250 million Box office $1.153 billion Captain
America : Civil War is a 2016 American superhero film based on the Marvel Comics character Captain America , produced by Marvel Studios and distributed by Walt
Disney Studios Motion Pictures .
</item 8>

<item 9>
Captain America: Civil War
The film 's Southeast Asia premiere was held on April 21 at the Marina Bay Sands resort in Singapore , while the European premiere took place at Vue Cinemas in
Westfield London on April 26 . The film was released internationally starting from April 27 , releasing in 61 countries its first weekend , including the United
Kingdom on April 29 . The North America release on May 6 , took place in over 4,200 theaters , of which 3,300 were in 3D , along with 378 IMAX theaters , 480
premium large - format , and 161 D - Box locations . Internationally , the film opened in 955 IMAX theaters , while South Korea saw Civil War open in an ``
unprecedented '' 1,989 theaters . In September 2014 , TNT acquired the cable broadcast rights for Captain America : Civil War to air two years after its
theatrical release . Marketing ( edit ) In early July 2015 , Marvel began a viral marketing campaign for Ant - Man featuring Leslie Bibb , reprising her role
from the Iron Man films as journalist Christine Everhart , reporting for the faux news program , WHIH Newsfront .
</item 9>

<item 10>
Captain America: Civil War
Filming locations in Germany included the Olympic Stadium in Berlin , and the Leipzig / Halle Airport in Schkeuditz . Additional filming also took place in
Puerto Rico and Norway , and was scheduled to in Iceland . Principal photography wrapped on August 22 , 2015 . Captain America : Civil War was the first film to
use IMAX 's digital 2D cameras , made in partnership with Arri , which was a customized version of the Arri Alexa 65 . According to Joe Russo , approximately
fifteen minutes of the film , the sequence where the two factions fight at Leipzig / Halle Airport , which was described as having `` incredible scale to it ''
and referred to by the directors as the `` splash panel '' , were shot with the cameras . Opaloch noted that there were discussions late in pre-production to
shoot the entire film with the Alexa 65 cameras . However , because of how close it was to the start of shooting as well as the idea being `` such a large -
scale undertaking '' , it was decided to only use them for the airport sequence , with the rest of film being shot on Arri Alexa XT digital cameras .
</item 10>
=== END ITEM id=f6e9922dfc1829ec ===

=== ITEM id=e876e0b9425a5b58 ===
Gold sentences and facts:
Sentence 0: `` Rip It Up '' was a 1983 single by Scottish post-punk band Orange Juice . It was the second single to be released from their 1982 album of the same name . The song became the band 's only UK top 40 success , reaching no . 8 in the chart . `` Rip It Up '' signalled a departure from the sound of the band 's earlier singles , with Chic - influenced guitars and using a synthesiser to create a more disco - oriented sound .
  Fact 0.0: 'Rip It Up' was a 1983 single by Scottish post-punk band Orange Juice.
  Fact 0.1: 'Rip It Up' was the second single released from Orange Juice's 1982 album of the same name.
  Fact 0.2: 'Rip It Up' became Orange Juice's only UK top 40 success.
  Fact 0.3: 'Rip It Up' reached no. 8 in the chart.
  Fact 0.4: 'Rip It Up' signalled a departure from the sound of the band's earlier singles.
  Fact 0.5: 'Rip It Up' had Chic-influenced guitars.
  Fact 0.6: 'Rip It Up' used a synthesiser to create a more disco-oriented sound.
Sentence 1: `` Rip It Up '' Single by Orange Juice from the album Rip It Up B - side `` Snake Charmer '' `` A Sad Lament '' Released February 1983 Format 7 '' 12 '' Recorded Berwick Street Studios , London Genre Post-punk Label Polydor Songwriter ( s ) Edwyn Collins Producer ( s ) Martin Hayles Orange Juice singles chronology `` I Ca n't Help Myself '' ( 1982 ) `` Rip It Up '' ( 1983 ) `` Flesh of My Flesh '' ( 1983 ) `` I Ca n't Help Myself '' ( 1982 ) `` Rip It Up '' ( 1983 ) `` Flesh of My Flesh '' ( 1983 )
  Fact 1.0: 'Rip It Up' is a single by Orange Juice from the album Rip It Up.
  Fact 1.1: The B-side of 'Rip It Up' is 'Snake Charmer'.
  Fact 1.2: The B-side of 'Rip It Up' is 'A Sad Lament'.
  Fact 1.3: 'Rip It Up' was released in February 1983.
  Fact 1.4: 'Rip It Up' was released in 7-inch and 12-inch formats.
  Fact 1.5: 'Rip It Up' was recorded at Berwick Street Studios, London.
  Fact 1.6: 'Rip It Up' is in the post-punk genre.
  Fact 1.7: 'Rip It Up' was released on the Polydor label.
  Fact 1.8: 'Rip It Up' was songwritten by Edwyn Collins.
  Fact 1.9: 'Rip It Up' was produced by Martin Hayles.
  Fact 1.10: Orange Juice's singles chronology lists 'I Ca n't Help Myself' (1982), 'Rip It Up' (1983) and 'Flesh of My Flesh' (1983).

Retrieved items:
<item 1>
Rip It Up (Orange Juice song)
Official Charts Company . Retrieved 26 May 2013 . External links ( edit ) Music video on YouTube ( hide ) Orange Juice Edwyn Collins James Kirk David McClymont
Steven Daly Malcolm Ross Zeke Manyika Studio albums You Ca n't Hide Your Love Forever Rip It Up Texas Fever The Orange Juice Singles `` Falling and Laughing ''
`` Rip It Up '' Compilations Ostrich Churchyard The Heather 's on Fire The Glasgow School Retrieved from ``
https://en.wikipedia.org/w/index.php?title=Rip_It_Up_(Orange_Juice_song)&oldid=751734896 '' Categories : 1983 singles Orange Juice ( band ) songs 1983 songs
Polydor Records singles Songs written by Edwyn Collins Hidden categories : Use dmy dates from May 2015 Use British English from May 2015 All articles with
unsourced statements Articles with unsourced statements from August 2013 Track listings with deprecated parameters Talk Contents About Wikipedia Add links This
page was last edited on 27 November 2016 , at 15 : 08 . About Wikipedia
</item 1>

<item 2>
Rip It Up (Orange Juice song)
Rip It Up ( Orange Juice song ) - wikipedia Rip It Up ( Orange Juice song ) Jump to : navigation , search `` Rip It Up '' Single by Orange Juice from the album
Rip It Up B - side `` Snake Charmer '' `` A Sad Lament '' Released February 1983 Format 7 '' 12 '' Recorded Berwick Street Studios , London Genre Post-punk
Label Polydor Songwriter ( s ) Edwyn Collins Producer ( s ) Martin Hayles Orange Juice singles chronology `` I Ca n't Help Myself '' ( 1982 ) `` Rip It Up '' (
1983 ) `` Flesh of My Flesh '' ( 1983 ) `` I Ca n't Help Myself '' ( 1982 ) `` Rip It Up '' ( 1983 ) `` Flesh of My Flesh '' ( 1983 ) `` Rip It Up '' was a 1983
single by Scottish post-punk band Orange Juice . It was the second single to be released from their 1982 album of the same name . The song became the band 's
only UK top 40 success , reaching no . 8 in the chart .
</item 2>

<item 3>
The Rolling Stones
Following the success of Some Girls , the band released their next album Emotional Rescue ( UK 1 ; US 1 ) in mid-1980 . During the recording sessions of the
album , a rift between Jagger and Richards was slowly beginning to form . Richards wanted to tour in summer or autumn of 1980 to promote the new album . Much to
his disappointment , Jagger declined . Emotional Rescue hit the top of the charts on both sides of the Atlantic and the title track reached No. 3 in the US .
The Rolling Stones performing in December 1981 In early 1981 , the group reconvened and decided to tour the US that year , leaving little time to write and
record a new album , as well as rehearse for the tour . That year 's resulting album , Tattoo You ( UK 2 ; US 1 ) , featured a number of outtakes , including
lead single `` Start Me Up '' , which reached No. 2 in the US and ranked No. 22 on Billboard 's Hot 100 year - end chart . Two songs ( `` Waiting on a Friend ''
( US No.
</item 3>

<item 4>
Rip It Up (Orange Juice song)
8 in the chart . `` Rip It Up '' signalled a departure from the sound of the band 's earlier singles , with Chic - influenced guitars and using a synthesiser to
create a more disco - oriented sound . The song was sampled in 2009 by British soul singer Beverley Knight on her song `` In Your Shoes '' from the album 100 %
. In 2014 , NME ranked it at number 216 in its list of The 500 Greatest Songs of All Time . It was also included by Pitchfork at number 157 in a list of The
Best 200 Songs of the 1980s . Contents ( hide ) 1 Recording and influences 2 Music video 3 UK single release 3.1 Track listing 4 Chart positions 5 References 6
External links Recording and influences ( edit ) Roland TB - 303 The song was recorded as part of the sessions for Orange Juice 's second studio album and would
go on to become the title track of said album .
</item 4>

<item 5>
Rip It Up (Orange Juice song)
The video then cuts back to the control room , this time with the band playing their instruments superimposed over it , before returning to more scenes of a
British city in torrential rain as the band walk around in scuba diving gear . The video finally cuts back to the band playing in a silver foil covered room ,
before superimposing them over a pile of random photographs . UK single release ( edit ) `` Rip It Up '' was released as a single in the UK in February 1983 .
The seven inch vinyl version of the single was available in three versions , a double pack including a second seven - inch and a fold out poster , along with
two versions of the standard release , initially with a silver injection moulded labels , and then subsequently with paper printed labels . The song was also
released on twelve inch vinyl , with extended versions of the title track and B - side . All versions were housed in a paper sleeve depicting a US P - 40
Warhawk fighter plane ( decorated with eyes and teeth ) partially submerged , tail first , in the sea , drawn by Edwyn Collins . Track listing ( edit ) Single 7
'' No .
</item 5>

<item 6>
Rip It Up (Orange Juice song)
`` Rip It Up '' ( Long Version ) Edwyn Collins 2 . `` A Sad Lament '' ( Long Version ) Edwyn Collins Chart positions ( edit ) Chart ( 1983 ) Peak position UK
Singles Chart 8 New Zealand Singles Chart 42 References ( edit ) Jump up ^ http://www.rocklistmusic.co.uk/nme_500_greatest_songs_2014.htm Jump up ^
http://pitchfork.com/features/lists-and-guides/9700-the-200-best-songs-of-the-1980s/?page=3 ^ Jump up to : `` Buzzcocks : Boredom / Orange Juice : Rip It Up ''
. Stylus Magazine . 25 August 2004 . Retrieved 26 May 2013 . ^ Jump up to : `` Orange Juice - Rip It Up 12 '' . Discogs . Retrieved 26 May 2013 . Jump up ^ ``
Orange Juice - Rip It Up '' . Discogs . Retrieved 26 May 2013 . Jump up ^ `` Orange Juice - Rip It Up '' . Discogs . Retrieved 26 May 2013 .
</item 6>

<item 7>
Rip It Up (Orange Juice song)
Track listing ( edit ) Single 7 '' No . Title Writer ( s ) Length 1 . `` Rip It Up '' Edwyn Collins 3 : 51 2 . `` Snake Charmer '' Malcolm Ross 4 : 43 Double 7
'' No . Title Writer ( s ) Length 1 . `` Rip It Up '' Edwyn Collins 2 . `` Snake Charmer '' Malcolm Ross 3 . `` Love Sick '' ( Live ) Edwyn Collins 4 . `` A Sad
Lament '' Edwyn Collins 12 '' No . Title Writer ( s ) Length 1 . `` Rip It Up '' ( Long Version ) Edwyn Collins 2 .
</item 7>

<item 8>
The Rolling Stones
He also encouraged the press to use provocative headlines such as `` Would you let your daughter marry a Rolling Stone ? '' , although Wyman says `` Our
reputation and image as the Bad Boys came later , completely there , accidentally ... ( Oldham ) never did engineer it . He simply exploited it exhaustively ''
. In a 1972 interview , Wyman stated `` We were the first pop group to break away from the whole Cliff Richard thing where the bands did little dance steps ,
wore identical uniforms and had snappy patter . '' A cover version of Chuck Berry 's `` Come On '' was the Rolling Stones ' first single , released on 7 June
1963 . The band refused to play it at live gigs , and Decca bought only one ad to promote the record . With Oldham 's direction , fan - club members bought
copies at record shops polled by the charts , helping `` Come On '' rise to No. 21 on the UK Singles Chart . Having a charting single gave the band entree to
play outside London , starting with a booking at the Outlook Club in Middlesbrough on 13 July , sharing the billing with the Hollies .
</item 8>

<item 9>
Rip It Up (Orange Juice song)
Discogs . Retrieved 26 May 2013 . ^ Jump up to : `` Orange Juice - Rip It Up ( Song ) '' . charts.org.nz . Retrieved 26 May 2013 . Jump up ^ `` Orange Juice -
Rip It Up '' . Discogs . Retrieved 26 May 2013 . Jump up ^ `` Orange Juice '' . Official Charts Company . Retrieved 26 May 2013 .
</item 9>

<item 10>
Rip It Up (Orange Juice song)
It marked a departure from their previous guitar - pop based material , instead utilising Chic style guitar - funk and a bubbling Roland TB - 303 synthesiser
bassline , becoming the first chart single to feature the instrument . The song also features a snatch of the guitar riff from `` Boredom '' , a song by
Buzzcocks that featured on their debut Spiral Scratch EP . The riff chimes briefly in , just as Collins namechecks the song in the lyrics claiming that `` ...
and my favourite song is entitled ' Boredom ' . '' Backing vocals on the song were provided by Paul Quinn , the lead singer of fellow Scottish band Bourgie
Bourgie , with whom Collins would later record a single in 1984 , a cover of the Velvet Underground song `` Pale Blue Eyes . '' Music video ( edit ) The video
opens with the band in a futuristic , but cheaply constructed , control room as they sing , dance and operate various controls . The band then watch themselves
on a monitor screen as they walk down a rainy British high street dressed in incongruous , brightly coloured summer clothes .
</item 10>
=== END ITEM id=e876e0b9425a5b58 ===
