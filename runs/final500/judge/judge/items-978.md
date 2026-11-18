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

=== ITEM id=d440ccd71f87f5c0 ===
Gold sentences and facts:
Sentence 0: In 1908 , the first five - day workweek in the United States was instituted by a New England cotton mill so that Jewish workers would not have to work on the Sabbath from sundown Friday to sundown Saturday . In 1926 , Henry Ford began shutting down his automotive factories for all of Saturday and Sunday . In 1929 , the Amalgamated Clothing Workers of America Union was the first union to demand a five - day workweek and receive it . After that , the rest of the United States slowly followed , but it was not until 1940 , when a provision of the 1938 Fair Labor Standards Act mandating a maximum 40 - hour workweek went into effect , that the two - day weekend was adopted nationwide .
  Fact 0.0: In 1908, the first five-day workweek in the United States was instituted by a New England cotton mill.
  Fact 0.1: The New England cotton mill instituted the five-day workweek so that Jewish workers would not have to work on the Sabbath from sundown Friday to sundown Saturday.
  Fact 0.2: In 1926, Henry Ford began shutting down his automotive factories for all of Saturday and Sunday.
  Fact 0.3: In 1929, the Amalgamated Clothing Workers of America Union was the first union to demand a five-day workweek.
  Fact 0.4: In 1929, the Amalgamated Clothing Workers of America Union received a five-day workweek.
  Fact 0.5: After that, the rest of the United States slowly followed.
  Fact 0.6: The two-day weekend was not adopted nationwide until 1940.
  Fact 0.7: The two-day weekend was adopted nationwide in 1940, when a provision of the 1938 Fair Labor Standards Act mandating a maximum 40-hour workweek went into effect.

Retrieved items:
<item 1>
Workweek and weekend
( February 4 , 2010 ) `` Shorter Workweek in a Tough Economy '' Archived November 28 , 2016 , at the Wayback Machine . Wisconsin Employment Law Letter (
hrhero.com ) Jump up ^ Rones et al. ( 1997 ) `` Trends in Hours of Work since the Mid-1970s '' Archived June 24 , 2016 , at the Wayback Machine . Monthly Labor
Review 120 ( 3 ) : 3 -- 12 Jump up ^ Coote , Anna ; Franklin , Jane ; Simms , Andrew ( February 2010 ) . `` 21 hours : Why a shorter working week can help us
all to flourish in the 21st century '' ( PDF ) . London : New Economics Foundation . ISBN 978 - 1 - 904882 - 70 - 1 . Archived from the original ( PDF ) on 9
February 2016 . Retrieved 18 October 2016 . Jump up ^ Stuart , H.
</item 1>

<item 2>
Workweek and weekend
In Ancient Rome , every eight days there was a nundinae . It was a market day , during which children were exempted from school and plebs ceased from work in
the field and came to the city to sell the produce of their labor or practice religious rites. . The French Revolutionary Calendar had ten - day weeks ( called
décades ) and allowed décadi , one out of the ten days , as a leisure day . In cultures with a four - day week , the three Sabbaths derives from the culture 's
main religious tradition : Friday ( Muslim ) , Saturday ( Jewish ) , and Sunday ( Christian ) . The present - day concept of the relatively longer ' week - end
' first arose in the industrial north of Britain in the early part of nineteenth century and was originally a voluntary arrangement between factory owners and
workers allowing Saturday afternoon off from 2pm in agreement that staff would be available for work sober and refreshed on Monday morning . The Oxford English
Dictionary traces the first use of the term weekend to the British magazine Notes and Queries in 1879 .
</item 2>

<item 3>
Graph query row:
entity: Institution of the first five-day workweek in the United States
date: 1908
relationship: RELATED_TO
connection: None
related: five-day workweek
related_date: None
</item 3>

<item 4>
Workweek and weekend
Retrieved 18 October 2016 . Jump up ^ Stuart , H. ( January 7 , 2012 ) `` Cut the working week to a maximum of 20 hours , urge top economists '' Archived
November 20 , 2012 , at the Wayback Machine . The Guardian Jump up ^ Schachter , H. ( February 10 , 2012 ) `` Save the world with a 3 - day work week ''
Archived May 5 , 2016 , at the Wayback Machine . Globe and Mail Jump up ^ Baker , D. ( January 27 , 2009 ) `` Pass the stimulus -- then help shorten the work
week '' Archived March 4 , 2016 , at the Wayback Machine . New York Daily News Jump up ^ Abate , T. ( July 11 , 2010 ) `` Get to Work : Want more jobs ? Shorten
the workweek '' Archived July 12 , 2010 , at the Wayback Machine .
</item 4>

<item 5>
Workweek and weekend
In 1926 , Henry Ford standardized on a five - day workweek , instead of the prevalent six days , without reducing employees ' pay . Hours worked stabilized at
about 49 per week during the 1920s , and during the Great Depression fell below 40 . During the Depression , President Herbert Hoover called for a reduction in
work hours in lieu of layoffs . Later , President Franklin Roosevelt signed the Fair Labor Standards Act of 1938 , which established a five - day , 40 - hour
workweek for many workers . The proportion of people working very long weeks has since risen , and the full - time employment of women has increased
dramatically . The New Economics Foundation has recommended moving to a 21 - hour standard workweek to address problems with unemployment , high carbon
emissions , low well - being , entrenched inequalities , overworking , family care , and the general lack of free time . The Center for Economic and Policy
Research states that reducing the length of the work week would slow climate change and have other environmental benefits . Around the world ( edit ) (
Countries listed alphabetically . Some countries have Saturday a normal school day .
</item 5>

<item 6>
Workweek and weekend
Of the remaining 50 % , 23 % have taken steps to reform their 6 - day workweek through such measures as reducing the working week from 6 days to 5.5 days .
United states ( edit ) The standard working week in the United States begins on Monday and ends on Friday , 40 hours per week , with Saturday and Sunday being
weekend days . However , in practice , only 42 % of employees work 40 - hour weeks . The average workweek for full - time employees is 47 hours . Most stores
are open for business on Saturday and often on Sunday as well , except in a few places where prohibited by law ( see Blue law ) . Increasingly , employers are
offering compressed work schedules to employees . Some government and corporate employees now work a 9 / 80 work schedule ( 80 hours over 9 days during a two -
week period ) -- commonly 9 hour days Monday to Thursday , 8 hours on one Friday , and off the following Friday . Jobs in healthcare , law enforcement ,
transportation , retail , and other service positions commonly require employees to work on the weekend or to do shift work .
</item 6>

<item 7>
Workweek and weekend
1 Soviet Union 3.17 Singapore 3.18 South Africa 3.19 Thailand 3.20 United States 3.21 Vietnam 4 See also 5 References History ( edit ) Part of a series on
Organised labour Labour movement ( show ) Timeline New Unionism Proletariat Social Movement Unionism Socialism Syndicalism Anarcho - syndicalism Labour rights (
hide ) Child labour Eight - hour day Five - day work week Annual leave Collective bargaining Minimum wage Occupational safety and health Trade unions ( show )
Trade unions by country Trade union federations International comparisons ITUC IWA WFTU Strike action ( show ) Chronological list of strikes General strike
Secondary action Sitdown strike Work - to - rule Labour parties ( show ) Australian British Dutch Irish Israeli New Zealand Norwegian List of Labour parties
Academic disciplines ( show ) Industrial relations Labour economics Labour history Labour law This section needs expansion . You can help by adding to it . (
September 2013 ) A continuous seven - day cycle that runs throughout history paying no attention whatsoever to the phases of the moon , having a fixed day of
rest , was probably first practiced in Judaism , dated to the 6th century BC at the latest .
</item 7>

<item 8>
Workweek and weekend
In 1908 , the first five - day workweek in the United States was instituted by a New England cotton mill so that Jewish workers would not have to work on the
Sabbath from sundown Friday to sundown Saturday . In 1926 , Henry Ford began shutting down his automotive factories for all of Saturday and Sunday . In 1929 ,
the Amalgamated Clothing Workers of America Union was the first union to demand a five - day workweek and receive it . After that , the rest of the United
States slowly followed , but it was not until 1940 , when a provision of the 1938 Fair Labor Standards Act mandating a maximum 40 - hour workweek went into
effect , that the two - day weekend was adopted nationwide . Over the succeeding decades , particularly in the 1940s , 1950s , and 1960s , an increasing number
of countries adopted either a Friday -- Saturday or Saturday -- Sunday weekend to harmonize with international markets .
</item 8>

<item 9>
Workweek and weekend
Most countries have adopted a two - day weekend , however , the days of the weekend differ according to religious tradition , i.e. either Thursday -- Friday ,
Friday -- Saturday , or Saturday -- Sunday , with the previous evening post-work often considered part of the weekend . Proposals have continued to be put
forward for further reductions in the number of days or hours worked per week , on the basis of predicted social and economic benefits . Contents ( hide ) 1
History 2 Length 3 Around the world 3.1 Australia 3.2 Brazil 3.3 Chile 3.4 China 3.4. 1 Hong Kong 3.5 Colombia 3.6 European Union members 3.6. 1 Austria 3.6. 2
Belgium 3.6. 3 Bulgaria 3.6. 4 Croatia 3.6. 5 Czech Republic 3.6. 6 Denmark 3.6. 7 Estonia 3.6. 8 Finland 3.6. 9 France 3.6. 10 Greece 3.6. 11 Hungary 3.6. 12
Ireland 3.6.
</item 9>

<item 10>
Workweek and weekend
In some Christian traditions , Sunday is the `` day of rest and worship '' . Jewish Shabbat or Biblical Sabbath lasts from sunset on Friday to the fall of full
darkness on Saturday ; as a result , the weekend in Israel is observed on Friday -- Saturday . Some Muslim - majority countries historically had a Thursday --
Friday or Friday -- Saturday weekend ; however , recently many such countries have shifted from Thursday -- Friday to Friday -- Saturday , or to Saturday --
Sunday . The Christian Sabbath was just one day each week , but the preceding day ( the Jewish Sabbath ) came to be taken as a holiday as well in the twentieth
century . This shift has been accompanied by a reduction in the total number of hours worked per week , following changes in employer expectations . The present
- day concept of the ' week - end ' first arose in the industrial north of Britain in the early part of nineteenth century and was originally a voluntary
arrangement between factory owners and workers allowing Saturday afternoon off from 2pm in agreement that staff would be available for work sober and refreshed
on Monday morning . The Amalgamated Clothing Workers of America Union was the first to successfully demand a five - day work week in 1929 .
</item 10>
=== END ITEM id=d440ccd71f87f5c0 ===

=== ITEM id=a076f12876ead410 ===
Gold sentences and facts:
Sentence 0: No . Title Featured guest Original release date `` It 's a Whole New Ball Game Now '' Barack Obama January 12 , 2018 ( 2018 - 01 - 12 ) Cold open : Letterman interviewing Obama on Late Show with David Letterman . On - location segment : Walking with Congressman John Lewis across the Edmund Pettus Bridge , site of the civil rights movement 's Bloody Sunday in 1965 . `` You Be the Newsman , I 'll Be Liz Taylor '' George Clooney February 9 , 2018 ( 2018 - 02 - 09 ) Cold open : Letterman and Clooney outside of LAX watching planes land and eating fast food . On - location segment : At the home of Nick Clooney meeting Iraqi refugee Hazim Avdal whom the Clooney family is sponsoring . `` You Know , She Has a Nobel Peace Prize '' Malala Yousafzai March 9 , 2018 ( 2018 - 03 - 09 ) Cold open : Letterman accompanies Yousafzai as she leads a tour group at Lady Margaret Hall , Oxford . On - location segment : Yousafzai continues to lead the tour group as Letterman tags along , the pair browse in the University of Oxford shop , and Letterman eats lunch with Yousafzai 's father . `` I Had a Paper Route Too '' Jay - Z April 6 , 2018 ( 2018 - 04 - 06 ) Cold open : None On - location segment : Letterman visits Shangri La Studio , speaks with music producer Rick Rubin and listens to Madison Ryann Ward recording . 5 `` TBA '' Tina Fey May 4 , 2018 6 `` TBA '' Howard Stern May 31 , 2018
  Fact 0.0: The table lists episode 1, titled 'It's a Whole New Ball Game Now', with featured guest Barack Obama, original release date January 12, 2018.
  Fact 0.1: The cold open for episode 1 has Letterman interviewing Obama on Late Show with David Letterman.
  Fact 0.2: The on-location segment for episode 1 has Letterman walking with Congressman John Lewis across the Edmund Pettus Bridge, site of the civil rights movement's Bloody Sunday in 1965.
  Fact 0.3: The table lists episode 2, titled 'You Be the Newsman, I'll Be Liz Taylor', with featured guest George Clooney, original release date February 9, 2018.
  Fact 0.4: The cold open for episode 2 has Letterman and Clooney outside of LAX watching planes land and eating fast food.
  Fact 0.5: The on-location segment for episode 2 is at the home of Nick Clooney meeting Iraqi refugee Hazim Avdal, whom the Clooney family is sponsoring.
  Fact 0.6: The table lists episode 3, titled 'You Know, She Has a Nobel Peace Prize', with featured guest Malala Yousafzai, original release date March 9, 2018.
  Fact 0.7: The cold open for episode 3 has Letterman accompanying Yousafzai as she leads a tour group at Lady Margaret Hall, Oxford.
  Fact 0.8: The on-location segment for episode 3 has Yousafzai continuing to lead the tour group while Letterman tags along.
  Fact 0.9: In the episode 3 on-location segment, the pair browse in the University of Oxford shop.
  Fact 0.10: In the episode 3 on-location segment, Letterman eats lunch with Yousafzai's father.
  Fact 0.11: The table lists episode 4, titled 'I Had a Paper Route Too', with featured guest Jay-Z, original release date April 6, 2018.
  Fact 0.12: The cold open for episode 4 is none.
  Fact 0.13: The on-location segment for episode 4 has Letterman visiting Shangri La Studio.
  Fact 0.14: In the episode 4 on-location segment, Letterman speaks with music producer Rick Rubin.
  Fact 0.15: In the episode 4 on-location segment, Letterman listens to Madison Ryann Ward recording.
  Fact 0.16: The table lists episode 5, titled 'TBA', with featured guest Tina Fey, original release date May 4, 2018.
  Fact 0.17: The table lists episode 6, titled 'TBA', with featured guest Howard Stern, original release date May 31, 2018.

Retrieved items:
<item 1>
*My Next Guest Needs No Introduction with David Letterman* is a Netflix talk show hosted by David Letterman. Each episode features an in-depth interview with
one guest, conducted both in a studio and outside it. Directed by Michael Bonfiglio, the English-language series is produced by RadicalMedia and Worldwide
Pants, with music by Paul Shaffer. Episodes run 50–56 minutes; the series premiered January 12, 2018.
</item 1>

<item 2>
My Next Guest Needs No Introduction with David Letterman
P.D. ( TBA ) Quicksand ( TBA ) Raising Dion ( TBA ) Ratched ( TBA ) The Dark Crystal : Age of Resistance ( TBA ) The Eddy ( TBA ) The Forgiving Earth ( TBA )
The Good Cop ( TBA ) The Green Beret 's Guide to Surviving the Apocalypse ( TBA ) The Haunting of Hill House ( TBA ) The Kominsky Method ( TBA ) The Order ( TBA
) The Politician ( TBA ) The Prince of Peoria ( TBA ) The Spy ( TBA ) The Witcher ( TBA ) Tidelands ( TBA ) Tuca & Bertie ( TBA ) Unbelievable ( TBA ) V - Wars
( TBA ) Wanderlust ( TBA ) Watership Down ( TBA ) Retrieved from `` https://en.wikipedia.org/w/index.
</item 2>

<item 3>
My Next Guest Needs No Introduction with David Letterman
The series consists of interviews with one guest per episode both inside and outside a studio setting . The first season will consist of six episodes and will
include interviews with Barack Obama , George Clooney , Malala Yousafzai , Jay - Z , Tina Fey , and Howard Stern . Obama was the first guest on the show , with
the first episode released on Netflix on January 12 , 2018 . Contents ( hide ) 1 Production 1.1 Development 1.2 Music 2 Episodes 3 Reception 4 References 5
External links Production ( edit ) Development ( edit ) On August 8 , 2017 , it was announced that Netflix had ordered a six - episode interview series starring
David Letterman . Each hour - long episode will feature Letterman conducting a longform conversation with a single guest and in - the - field segments , in
which he will explore topics on his own . On January 5 , 2018 , Netflix released further information about the series .
</item 3>

<item 4>
Netflix’s *My Next Guest Needs No Introduction* marked David Letterman’s return to television with six hour-long episodes combining in-studio interviews and
field segments. Announced in August 2017, the first season featured Barack Obama (premiered January 12, 2018), George Clooney, Malala Yousafzai, Jay-Z, Tina
Fey, and Howard Stern. The season received generally favorable reviews: 80% on Rotten Tomatoes and 70/
</item 4>

<item 5>
My Next Guest Needs No Introduction with David Letterman
Washington Post . Retrieved January 13 , 2018 . Jump up ^ Leight , Elias ( February 2 , 2018 ) . `` Paul Shaffer on Writing Letterman 's Netflix Theme Song &
the Value of Humor in Pop '' . Billboard . Retrieved February 9 , 2018 . Jump up ^ Carr , Mary Kate ( April 24 , 2018 ) . `` What 's on Netflix in May : The
best shows and movies to watch this month '' . Entertainment Weekly . Retrieved April 24 , 2018 . Jump up ^ Douglas , Nick ( April 24 , 2018 ) . `` Here 's What
's Coming And Going From Netflix In May 2018 '' . Lifehacker . Retrieved April 24 , 2018 . Jump up ^ `` My Next Guest Needs No Introduction With David Letterman
: Season 1 - Rotten Tomatoes '' . Rotten Tomatoes . Retrieved January 13 , 2018 .
</item 5>

<item 6>
My Next Guest Needs No Introduction with David Letterman
My Next Guest Needs No Introduction with David Letterman - wikipedia My Next Guest Needs No Introduction with David Letterman Jump to : navigation , search My
Next Guest Needs No Introduction with David Letterman Genre Talk show Directed by Michael Bonfiglio Presented by David Letterman Composer ( s ) Paul Shaffer
Country of origin United States Original language ( s ) English No. of seasons No. of episodes 4 ( list of episodes ) Production Executive producer ( s ) Justin
Wilkes Dave Sirulnick Jon Kamen Tom Keany Producer ( s ) Mary Barclay Cinematography Declan Quinn Editor ( s ) Jesse Gordon Edward A. Bishop Paul Little Running
time 50 -- 56 minutes Production company ( s ) RadicalMedia Worldwide Pants Inc . Release Original network Netflix Original release January 12 , 2018 ( 2018 -
01 - 12 ) -- present External links Official website My Next Guest Needs No Introduction with David Letterman is an American web television talk show hosted by
David Letterman . The series consists of interviews with one guest per episode both inside and outside a studio setting .
</item 6>

<item 7>
My Next Guest Needs No Introduction with David Letterman
wikipedia.org/w/index.php?title=My_Next_Guest_Needs_No_Introduction_with_David_Letterman&oldid=838074810 '' Categories : 2010s American comedy television series
2018 American television series debuts David Letterman English - language television programs Netflix original programming Hidden categories : Pages using
infobox television with editor parameter Episode lists with unformatted air dates Talk Contents About Wikipedia Italiano Nederlands Português Edit links This
page was last edited on 24 April 2018 , at 19 : 28 . About Wikipedia
</item 7>

<item 8>
Netflix announced that *My Next Guest Needs No Introduction with David Letterman* would premiere January 12, 2018, with Barack Obama—his first interview since
leaving office—as its first guest. Season-one guests also included George Clooney, Malala Yousafzai, Jay-Z, Tina Fey, and Howard Stern. Paul Shaffer composed
the theme and score, drawing on Muscle Shoals soul and using piano, organ, and bass. Obama’s episode included Letterman walking the Edmund
</item 8>

<item 9>
My Next Guest Needs No Introduction with David Letterman
`` I Had a Paper Route Too '' Jay - Z April 6 , 2018 ( 2018 - 04 - 06 ) Cold open : None On - location segment : Letterman visits Shangri La Studio , speaks
with music producer Rick Rubin and listens to Madison Ryann Ward recording . 5 `` TBA '' Tina Fey May 4 , 2018 6 `` TBA '' Howard Stern May 31 , 2018 Reception
( edit ) The first season of My Next Guest Needs No Introduction has been met with a positive response from critics . On the review aggregation website Rotten
Tomatoes , the first season holds a 80 % approval rating with an average rating of 6.4 out of 10 based on 25 reviews . Metacritic , which uses a weighted
average , assigned the season a score of 70 out of 100 based on 13 reviews , indicating `` generally favorable reviews '' . References ( edit ) Jump up ^
Andreeva , Nellie ( August 8 , 2017 ) . `` David Letterman Returns To Television With Netflix Series '' . Deadline .
</item 9>

<item 10>
My Next Guest Needs No Introduction with David Letterman
Rotten Tomatoes . Retrieved January 13 , 2018 . Jump up ^ `` My Next Guest Needs No Introduction With David Letterman : Season 1 - Metacritic '' . Metacritic .
Retrieved January 13 , 2018 . External links ( edit ) Official website David Letterman Television hosting The David Letterman Show ( 1980 ) Late Night with
David Letterman ( 1982 -- 1993 ) Late Show with David Letterman ( 1993 -- 2015 ) My Next Guest Needs No Introduction with David Letterman ( 2018 ) Companies The
Letterman Foundation for Courtesy and Grooming Rahal Letterman Lanigan Racing Worldwide Pants Related Dorothy Mengering Netflix original series Current Since
2013 House of Cards Orange Is the New Black Since 2014 BoJack Horseman Since 2015 Unbreakable Kimmy Schmidt Chef 's Table Daredevil Grace and Frankie Residue
Sense8 Club de Cuervos Project Mc Dinotrux Narcos Master of None Jessica Jones F Is for Family Making a Murderer Real Rob Since 2016 Degrassi : Next Class
Fuller House Lego Friends : The Power of Friendship Flaked
</item 10>
=== END ITEM id=a076f12876ead410 ===

=== ITEM id=7d0e4b3a8e364548 ===
Gold sentences and facts:
Sentence 0: Sinéad ( / ʃɪˈneɪd / shi - NADE ; Irish pronunciation : ( ˈʃɪnjeːd̪ɣ ) or ( ʃɪˈnjeːd̪ɣ ) ) , is an Irish feminine name . It is derived from the French Jeanette , which is cognate to the English Janet , itself a feminine form of the Hebrew Yohannan , `` God forgave / God gratified '' . In English , Sinéad is also commonly spelled Sinead . The name is generally translated into English as either Jane or Jennifer , or as the female version of the Scottish name Jean .
  Fact 0.0: Sinéad is an Irish feminine name.
  Fact 0.1: Sinéad is pronounced /ʃɪˈneɪd/, shi-NADE, in English.
  Fact 0.2: Sinéad has an Irish pronunciation of /ˈʃɪnjeːd̪ɣ/ or /ʃɪˈnjeːd̪ɣ/.
  Fact 0.3: Sinéad is derived from the French name Jeanette.
  Fact 0.4: Jeanette is cognate to the English name Janet.
  Fact 0.5: Janet is itself a feminine form of the Hebrew name Yohannan.
  Fact 0.6: Yohannan means 'God forgave' or 'God gratified'.
  Fact 0.7: Sinéad is also commonly spelled Sinead in English.
  Fact 0.8: The name Sinéad is generally translated into English as Jane or Jennifer.
  Fact 0.9: The name Sinéad is generally translated into English as the female version of the Scottish name Jean.

Retrieved items:
<item 1>
I before E except after C
/ eɪ / FACE With eigh spelling : eight , freight , heigh - ho * , inveigh , neigh , neighbo ( u ) r , sleigh , weigh Others : abseil , beige , capoeira ,
cleidoic , deign , dreidel , feign , feint , geisha , glei , greige , greisen , heinous * , inveigle * , nonpareil * , obeisance * , peignoir * , reign , rein ,
seiche , seidel , seine , sheikh * , skein , surveillance , veil , vein . ( While Carney says this sound is never spelled ie , the last vowel in lingerie * is
often the FACE vowel . ) . / ɛər / SQUARE heir , their . ( This sound is never spelled ier ) With other sounds ( edit ) These are exceptions to the basic and ``
long a '' versions of the rhyme , but not to the `` long e '' version . This is a dynamic list and may never be able to satisfy particular standards for
completeness . You can help by expanding it with reliably sourced entries .
</item 1>

<item 2>
Sinéad
a character on the TV series Hollyoaks , played by Stephanie Davis Sinead Starling is a fictional character from The 39 Clues . Sinéad Ní Shúilleabháin , a
fictional character in The Wind That Shakes the Barley See also ( edit ) Siobhán , feminine Irish - language name ultimately derived from the same Hebrew name
References ( edit ) ( hide ) Gaelic given names Native masculine names Aodh Aodhán Aonghus Art Breandán Breasal Brian Cailean Cainnech Caoimhín Cathal Cearbhall
Cian Cianán Ciarán Cinnéididh Colm Conall Conán Conchobhar Cormac Criofan Dáire Déaglán Diarmaid Domhnall Donn Donnchadh Eachann Éibhear Eochaidh Eóghan
Fearghal Fearghus Féilim Fiach Fínghin Fionn Fionnbharr Fionnlagh Fionntán Flann Garaidh Iarlaith Lorcan Lughaidh Muircheartach Muiredach Muirgheas Murchadh
</item 2>

<item 3>
I before E except after C
Less strict interpretations admit as exceptions those words where eir , not preceded by c , represents the vowel of NEAR ( / ɪər / ) . Some categories of
exception : Many proper names , often because they are adopted from other languages . Fowler says the rule `` is useless with proper names '' ; Carney says ``
As one might expect of any rule , there are likely to be even more exceptions in names , many of which are Scottish '' : forenames and surnames Keith , Neil ,
Sheila , Stein , etc . placenames Leith , Keighley , Rheims , Raleigh , etc . Eid in the names of Muslim holidays ( Eid ul - Fitr , Eid al - Adha , etc . )
others like Cassiopeia Chemical names ending in - ein or - eine ( caffeine , casein , codeine , phthalein , protein , etc . ) . Here - ein ( e ) was originally
pronounced as two syllables / iː. ɪn / Scottish English words ( deil , deid , weill , etc . )
</item 3>

<item 4>
DNA
Sense and antisense Further information : Sense ( molecular biology ) A DNA sequence is called `` sense '' if its sequence is the same as that of a messenger
RNA copy that is translated into protein . The sequence on the opposite strand is called the `` antisense '' sequence . Both sense and antisense sequences can
exist on different parts of the same strand of DNA ( i.e. both strands can contain both sense and antisense sequences ) . In both prokaryotes and eukaryotes ,
antisense RNA sequences are produced , but the functions of these RNAs are not entirely clear . One proposal is that antisense RNAs are involved in regulating
gene expression through RNA - RNA base pairing . A few DNA sequences in prokaryotes and eukaryotes , and more in plasmids and viruses , blur the distinction
between sense and antisense strands by having overlapping genes . In these cases , some DNA sequences do double duty , encoding one protein when read along one
strand , and a second protein when read in the opposite direction along the other strand .
</item 4>

<item 5>
Sinéad
Sinéad - wikipedia Sinéad Jump to : navigation , search Sinéad ( / ʃɪˈneɪd / shi - NADE ; Irish pronunciation : ( ˈʃɪnjeːd̪ɣ ) or ( ʃɪˈnjeːd̪ɣ ) ) , is an Irish
feminine name . It is derived from the French Jeanette , which is cognate to the English Janet , itself a feminine form of the Hebrew Yohannan , `` God forgave
/ God gratified '' . In English , Sinéad is also commonly spelled Sinead . The name is generally translated into English as either Jane or Jennifer , or as the
female version of the Scottish name Jean . Sinéad may refer to : Contents ( hide ) 1 People 2 Fiction 3 See also 4 References People ( edit ) Sports Sinéad
Cahalan , camogie player Sinead Jennings , rower Sinead Kerr , ice dancer Sinéad Millea , former camogie player Sinead Miller , cyclist Sinead Russell , Olympic
Swimmer Music `` Sinéad '' ,
</item 5>

<item 6>
Sinéad
Liam Maghnus Raghnall Risteárd Roibeárd Séarlas Somhairle Tiobóid Tormond Ualtar Ùisdean Bible - derived masculine names Abracham Adhamh Árón Biniáimin Daibhí
Dainéal Eafráim Gaibrial Íosac Maitiú Maois Mícheál ( Maidhc ) Nóe Seán Séamus Seosamh Síomón Solamh Tomás Bible - derived feminine names Abigeál Ánna Eábha
Eilís Léá Madailéin Máire Mallaidh Náoimí Ráichéal Ríobhca Rút Sinéad Siobhán Seacailín Seosaimhín Latin / Greek - derived masculine names Aindréas Antóin
Críostóir Damhnaic Pádraig Proinsias Giolla Críost Gréagóir Labhrás León Máirtín Nioclás Nollaig Peadar Pilib Pól Seoirse Stíofán Téodóir Uinseann See also
Goidelic languages
</item 6>

<item 7>
In the Pines
In variants in which the song describes a confrontation , the person being challenged is always a woman . The folk version by the Kossoy Sisters asks , ``
Little girl , little girl , where 'd you stay last night ? Not even your mother knows . '' The reply to the question , `` Where did you get that dress / And
those shoes that are so fine ? '' from one version is , `` From a man in the mines / Who sleeps in the pines . '' The theme of a woman being caught doing
something she should not is thus also common to many variants . One variant , performed in the early twentieth century by the Ellison clan ( Ora Ellison ,
deceased ) in Lookout Mountain , Georgia , tells of a young Georgia girl who flees to the pines after being raped . Her rapist , a male soldier , is later
beheaded by the train . Some versions of the song also reference the Great Depression , with the `` black girl '' being a hobo on the move from the police , who
witnesses the murder of her father while train - jumping . She hides from this by sleeping in the pines , in the cold .
</item 7>

<item 8>
Sinéad
Lughaidh Muircheartach Muiredach Muirgheas Murchadh Niall Oisín Rían Rónán Ruairí Searbhreathach Tadhg Toirdelbach Native feminine names Aíbinn Aifric Áine
Aisling Aoife Barrdhubh Bébinn Bríd Cacht Caoimhe Ciara Clíona Clodagh Cobhlaith Dearbhail Dearbhfhorghaill Deirbhile Deirdre Dubhchobhlaigh Eithne Éadaoin
Feidhelm Fíona Fionnuala Forbflaith Fíneamhain Gormflaith Gráinne Íde Lasairfhíona Meabh Mór Muirgheal Muirenn Niamh Nuala Órlaith Sadhbh Saoirse Síthmaith
Sláine Sorcha Úna Germanic - derived masculine names Amhlaoibh Annraoi Dubhghall Éamonn Gearóid Gofraid Ímar Lachlann Liam Maghnus Raghnall Risteárd Roibeárd
Séarlas
</item 8>

<item 9>
Sinéad
Stíofán Téodóir Uinseann See also Goidelic languages Irish language Scottish Gaelic language Manx language Irish name List of Scottish Gaelic given names
Scottish Gaelic name Irish genealogy This page or section lists people that share the same given name . If an internal link led you here , you may wish to
change that link to point directly to the intended article . Retrieved from `` https://en.wikipedia.org/w/index.php?title= Sinéad&oldid = 792745813 ''
Categories : Given names Irish feminine given names Irish - language feminine given names Hidden categories : All set index articles Talk Contents About
Wikipedia Čeština Deutsch Español Français Italiano 日本 語 Edit links This page was last edited on 28 July 2017 , at 11 : 52 . Text is available under the
Creative Commons Attribution - ShareAlike License ; additional terms may apply . By using this site , you agree to the Terms of Use and Privacy Policy .
Wikipedia ® is a registered trademark of the Wikimedia Foundation , Inc. , a non-profit organization . About Wikipedia
</item 9>

<item 10>
Sinéad
cyclist Sinead Russell , Olympic Swimmer Music `` Sinéad '' , a song by Within Temptation from their album The Unforgiving , and the name of a character in the
album 's tie - in comic book Sinead Harnett , singer / songwriter Sinéad Lohan , singer / songwriter Sinéad Madden , singer / songwriter Sinéad Mulvey , singer
, air hostess Sinéad O'Carroll , singer with Irish pop band B * Witched Sinéad O'Connor , singer / songwriter Sinéad Quinn , recording artist , reality show
contestant Film and television Sinéad Cusack , actress Sinead Desmond , TV presenter Sinead Keenan , actress Sinead Matthews , actress Sinéad Moynihan , model
and actress Sinéad Noonan , model and actress Other Sinéad de Valera , children 's book author Sinéad Morrissey , poet Sinéad Ní Loideáin , presenter Sinéad Ní
Neachtain , magazine editor Sinéad Sheppard , dancer Fiction ( edit ) Sinead O'Connor ( Hollyoaks ) , a character on the TV series Hollyoaks ,
</item 10>
=== END ITEM id=7d0e4b3a8e364548 ===

=== ITEM id=90dcf672ff7770f7 ===
Gold sentences and facts:
Sentence 0: Republic of Doyle is a Canadian comedy - drama television series set in St. John 's , Newfoundland and Labrador which aired on CBC Television from January 6 , 2010 to December 10 , 2014 .
  Fact 0.0: Republic of Doyle is a Canadian comedy-drama television series.
  Fact 0.1: Republic of Doyle is set in St. John's, Newfoundland and Labrador.
  Fact 0.2: Republic of Doyle aired on CBC Television.
  Fact 0.3: Republic of Doyle aired from January 6, 2010 to December 10, 2014.

Retrieved items:
<item 1>
Republic of Doyle
Sloan gets into trouble and Jake must bail her out by making a deal with Vick Saul . Even though she was not cleared by her therapist after having a panic
attack on duty , Leslie lends a hand to the Doyle clan to try and clear Jake 's name before it is too late . When Leslie 's past disobedience on the force comes
to light , she must work harder than ever to save her job . Jake 's murder conviction comes front and center as the season comes to a final close with all the
Doyle family trying to clear his name . Episodes ( edit ) Main article : List of Republic of Doyle episodes Season Episodes Originally aired First aired Last
aired 12 January 6 , 2010 ( 2010 - 01 - 06 ) April 7 , 2010 ( 2010 - 04 - 07 ) 13 January 12 , 2011 ( 2011 - 01 - 12 ) April 6 , 2011 ( 2011 - 04 - 06 ) 13
January 11 , 2012 ( 2012 - 01 - 11 ) April 4 ,
</item 1>

<item 2>
Republic of Doyle
The character name is evocative of the character name , Bobby Crocker , and the actor who played him , Kevin Dobson , in the classic television series Kojak (
1973 -- 78 ) , starring Telly Savalas . Production ( edit ) The show is filmed in and around St. John 's , Newfoundland and Labrador . The show 's star , Allan
Hawco , is also executive producer / showrunner / co-creator and head writer of the show . Other producers include John Vatcher ( exec producer ) , Rob Blackie
, Perry Chafe ( co-creator and writer ) and Michael Levine . In 2013 , Republic of Doyle aired a fictional crossover with another CBC Television crime drama ,
Murdoch Mysteries . With the two shows set over 100 years apart , Hawco appeared on the November 25 , 2013 episode of Murdoch Mysteries as Jacob Doyle , a 19th
- century ancestor of Jake Doyle , while Yannick Bisson appeared on the January 29 , 2014 episode of Republic of Doyle as Detective Bill Murdoch , a 21st -
century descendant of his regular character Detective William Murdoch .
</item 2>

<item 3>
Republic of Doyle
Season synopses ( edit ) Season one ( edit ) Premiering on January 6 , 2010 , the show introduces Allan Hawco as Jake Doyle and Sean McGinley as Mal Doyle as a
father and son pair of private investigators . The debut show also reveals relationship complications with Jake and his ex-wife Nikki and his new infatuation :
Leslie , the Inspector who works at the Royal Newfoundland Constabulary . While Jake tries to improve his relationship with Leslie , he discovers he may have
impregnated his ex-wife , but soon discovers the positive pregnancy test he found in her garbage can did not belong to her . After he realizes he is free from
the burden of having to deal with child - support , his relationship with Leslie develops as Nikki 's relationship with her new boyfriend grows . However ,
Leslie soon begins to decide whether her relationship with Jake will prove beneficial in the future . As Jake finalizes his divorce with Nikki , his
relationship status with Leslie improves to good friends . Yet Leslie 's focus quickly shifts to her new title as Sergeant after being promoted to the position
.
</item 3>

<item 4>
Republic of Doyle
On September 6 , 2016 , eOne will release Republic of Doyle - The Complete Series on DVD in Canada . Acorn Media has released the first two seasons on DVD in
the USA . In Region 4 , Beyond Home Entertainment has released the first two seasons on DVD in Australia . DVD name Ep # Release dates Region 1 ( CAN ) Region 4
The Complete First Season 12 November 16 , 2010 May 4 , 2011 The Complete Second Season 13 July 19 , 2011 April 4 , 2012 The Complete Third Season 13 November
20 , 2012 TBA The Complete Fourth Season 13 October 1 , 2013 TBA The Complete Fifth Season 16 September 30 , 2014 TBA The Complete Sixth Season 10 April 21 ,
2015 TBA The Complete Series 77 September 6 ,
</item 4>

<item 5>
Republic of Doyle
2015 TBA The Complete Series 77 September 6 , 2016 TBA International broadcasters ( edit ) Country Channel Premiere date South Africa Fox Entertainment 1 August
2010 United Kingdom Alibi 27 October 2010 Australia ABC2 ( season one ) 13th Street ( season two -- ) 6 November 2010 Brazil Globosat HD 11 November 2010
Estonia Fox Crime 11 September 2010 Serbia Fox Crime 14 October 2010 Bulgaria Fox Crime 4 December 2010 Poland TVP2 25 December 2012 Germany 13th Street 16
October 2013 United States CW Every Saturday and Sunday Netherlands 13th Street Until Friday 13 March 2015 , daily Monday - Friday Republic of Doyle is aired in
the USA on various stations via syndication . References ( edit ) Jump up ^ `` CBC winter season brings 3 new shows '' . cbc.ca , November 25 , 2009 . Jump up ^
`` Full Cast & Crew . '' IMDb . IMDb.com , n.d. Web . 17 Feb. 2014 . Jump up ^ `` Republic of Doyle renewed for 6th season '' .
</item 5>

<item 6>
Republic of Doyle
Republic of Doyle - wikipedia Republic of Doyle Jump to : navigation , search Republic of Doyle Genre Comedy - drama Police procedural Created by Allan Hawco
Malcolm MacRury Perry Chafe Starring Allan Hawco Seán McGinley Lynda Boyd Krystin Pellerin Mark O'Brien Marthe Bernard Bob Cole Opening theme `` Oh Yeah '' by
Great Big Sea Original language ( s ) English No. of seasons 6 No. of episodes 77 ( list of episodes ) Production Executive producer ( s ) Allan Hawco , John
Vatcher , Michael Levine Location ( s ) St. John 's , Newfoundland and Labrador , Canada Cinematography Malcolm Cross Running time 44 minutes Production company
( s ) Take the Shot Productions Fireworks ( seasons 1 - 2 ) Content Film ( season 3 - 6 ) Distributor Trifecta Entertainment & Media ( United States ) Release
Original network CBC Television Picture format 1080i ( HDTV ) Original release January 6 , 2010 -- December 10 , 2014 External links Website Republic of Doyle
is a Canadian comedy - drama television series set in St.
</item 6>

<item 7>
Republic of Doyle
Jump up ^ Newsome , Brad ( 6 January 2012 ) . `` Republic of Doyle , Friday , January 6 '' . The Sydney Morning Herald . Retrieved 17 January 2012 . Jump up ^
Daniel MacEachern ( 2013 - 04 - 17 ) . `` ' Republic of Doyle ' headed for U.S. '' The Telegram . Retrieved 2014 - 11 - 21 . External links ( edit ) Television
in Canada portal Official website Republic of Doyle on IMDb Retrieved from `` https://en.wikipedia.org/w/index.php?title=Republic_of_Doyle&oldid=837889899 ''
Categories : 2010 Canadian television series debuts CBC Television shows Culture of St.
</item 7>

<item 8>
Republic of Doyle
Contents ( hide ) 1 Cast 1.1 Main 1.2 Recurring 2 Characters 3 Production 4 Season synopses 4.1 Season One 4.2 Season Two 4.3 Season Three 4.4 Season Four 4.5
Season Five 4.6 Season Six 5 Episodes 6 Reception 6.1 Ratings 7 Awards and nominations 7.1 Canadian Cinema Editors Awards 7.2 Canadian Comedy Awards 7.3
Canadian Screen Awards 7.4 Gemini Awards 8 DVD releases 9 International broadcasters 10 References 11 External links Cast ( edit ) Main ( edit ) Allan Hawco --
Jake Doyle ( Seasons 1 -- 6 ) Seán McGinley -- Malachy Doyle ( Seasons 1 -- 6 ) Lynda Boyd -- Rose Miller ( Seasons 1 -- 6 ) Krystin Pellerin -- Leslie Bennett
( Seasons 1 -- 6 ) Rachel Wilson -- Dr. Nikki Renholds ( Main , Season 1 ; Recurring , Seasons 2 -- 6 ) Mark O'Brien -- Desmond `` Des '' Courtney ( Seasons 1
-- 6 ) Marthe Bernard -- Katrina `` Tinny '' Doyle (
</item 8>

<item 9>
Republic of Doyle
Jump up ^ `` Republic of Doyle , Season 2 '' . Amazon.com . Retrieved 2017 - 10 - 04 . Jump up ^ `` REPUBLIC OF DOYLE - SEASON 1 Television Series Drama Comedy
'' . Beyondhomeentertainment.com.au . Retrieved 2012 - 11 - 09 . Jump up ^ `` REPUBLIC OF DOYLE - SEASON 2 Television Series Drama Comedy '' .
Beyondhomeentertainment.com.au . Retrieved 2012 - 11 - 09 . Jump up ^ `` Republic of Doyle '' . TVSA. 2010 - 08 - 01 . Retrieved 2010 - 08 - 01 . Jump up ^ ``
UKTV enters Republic of Doyle '' ( Press release ) . C21 Media . 2010 - 08 - 26 . Retrieved 2010 - 08 - 26 . Jump up ^ `` ABC2 Programming Airdate : Republic of
Doyle ( episode one ) '' . ABC Television Publicity. 2010 - 11 - 20 . Jump up ^ Newsome , Brad ( 6 January 2012 ) .
</item 9>

<item 10>
Republic of Doyle
Rose gets in trouble with the police for carrying a gun . Tinny is bothered by the quickly growing relationship between Des and Chandra , a university student
of medicine . Kathleen , a newly introduced character that serves as Jake 's sister , is helped with her mothering issues by Rose . The two begin to progress in
their friendship . Des finds himself in a precarious position -- choosing between Tinny and Chandra . He decides to break up with Chandra and pursue a romance
with Tinny . Kathleen faces parenting issues at home . Des attempts to impress Jake by helping him in an investigation using his various connections , while
Allison jeopardizes the safety of the Doyle household with a mistake from her past . Mal becomes tied up in complications with his former wife while Tinny
obsesses over her death and Kathleen experiences troubles in their relations as a result of a poor decision in Kathleen 's history . George , an unappreciated
individual in the Republic , bothers Kathleen by scamming her , but Jake comes to the rescue and saves her from his tricks . The season ends with Tinny
discovering the identity of her father , which is not revealed to the viewers .
</item 10>
=== END ITEM id=90dcf672ff7770f7 ===

=== ITEM id=01f47b3891dd87d7 ===
Gold sentences and facts:
Sentence 0: John Brown 's raid on Harper 's Ferry ( also known as John Brown 's raid or The raid on Harper 's Ferry ) was an effort by armed abolitionist John Brown to initiate an armed slave revolt in 1859 by taking over a United States arsenal at Harpers Ferry , Virginia . Brown 's party of 22 was defeated by a company of U.S. Marines , led by First Lieutenant Israel Greene . Colonel Robert E. Lee was in overall command of the operation to retake the arsenal . John Brown had originally asked Harriet Tubman and Frederick Douglass , both of whom he had met in his transformative years as an abolitionist in Springfield , Massachusetts , to join him in his raid , but Tubman was prevented by illness and Douglass declined , as he believed Brown 's plan would fail .
  Fact 0.0: John Brown's raid on Harper's Ferry is also known as John Brown's raid.
  Fact 0.1: John Brown's raid on Harper's Ferry is also known as The raid on Harper's Ferry.
  Fact 0.2: John Brown's raid on Harper's Ferry was an effort by armed abolitionist John Brown to initiate an armed slave revolt in 1859.
  Fact 0.3: The raid aimed to start the armed slave revolt by taking over a United States arsenal at Harpers Ferry, Virginia.
  Fact 0.4: Brown's party of 22 was defeated by a company of U.S. Marines.
  Fact 0.5: The company of U.S. Marines that defeated Brown's party was led by First Lieutenant Israel Greene.
  Fact 0.6: Colonel Robert E. Lee was in overall command of the operation to retake the arsenal.
  Fact 0.7: John Brown had originally asked Harriet Tubman to join him in the raid.
  Fact 0.8: John Brown had originally asked Frederick Douglass to join him in the raid.
  Fact 0.9: John Brown had met Harriet Tubman and Frederick Douglass during his transformative years as an abolitionist in Springfield, Massachusetts.
  Fact 0.10: Harriet Tubman was prevented from joining the raid by illness.
  Fact 0.11: Frederick Douglass declined to join the raid.
  Fact 0.12: Frederick Douglass believed Brown's plan would fail.
Sentence 1: By 3 : 30 that afternoon , President James Buchanan ordered a company of U.S. Marines ( the only government troops in the immediate area ) to march on Harpers Ferry under the command of Brevet Colonel Robert E. Lee , lieutenant colonel of the 2nd U.S. Cavalry Regiment. Lee had been on leave from his regiment , stationed in Texas , when he was hastily called to lead the detachment and had to command it while wearing his civilian clothes .
  Fact 1.0: By 3:30 that afternoon, President James Buchanan ordered a company of U.S. Marines to march on Harpers Ferry.
  Fact 1.1: The Marines were the only government troops in the immediate area.
  Fact 1.2: The Marines marched on Harpers Ferry under the command of Brevet Colonel Robert E. Lee.
  Fact 1.3: Robert E. Lee was lieutenant colonel of the 2nd U.S. Cavalry Regiment.
  Fact 1.4: Lee had been on leave from his regiment when he was hastily called to lead the detachment.
  Fact 1.5: Lee's regiment was stationed in Texas.
  Fact 1.6: Lee had to command the detachment while wearing his civilian clothes.

Retrieved items:
<item 1>
John Brown's raid on Harpers Ferry
The Kennedy Farmhouse served as `` barracks , arsenal , supply depot , mess hall , debate club , and home '' . It was very crowded and life there was tedious .
Brown was worried about arousing neighbors ' suspicions . As a result , the raiders had to stay indoors during the daytime , without much to do but study ,
drill , argue politics , discuss religion , and play cards and checkers . Brown 's daughter - in - law Martha served as cook and housekeeper . His daughter
Annie served as lookout . Brown wanted women at the farm , to prevent suspicions of a large all - male group . The raiders went outside at night to drill and
get fresh air . Thunderstorms were welcome since they concealed noise from Brown 's neighbors . Brown did not plan to have a sudden raid and escape to the
mountains . Rather , he intended to use those rifles and pikes he captured at the arsenal , in addition to those he brought along , to arm rebellious slaves
with the aim of striking terror in the slaveholders in Virginia . He believed that on the first night of action , 200 - 500 black slaves would join his line .
</item 1>

<item 2>
John Brown's raid on Harpers Ferry
John Brown 's raid on Harpers Ferry - wikipedia John Brown 's raid on Harpers Ferry Jump to : navigation , search John Brown 's Raid on Harper 's Ferry Part of
pre-Civil War conflicts Harper 's Weekly illustration of U.S. Marines attacking John Brown 's `` Fort '' Date October 16 -- 18 , 1859 Location Harpers Ferry , (
West ) Virginia Result U.S. victory Belligerents United States Abolitionist Insurgents Commanders and leaders Robert E. Lee Israel Greene Jeb Stuart John Brown
Strength 88 U.S. Marines Unknown number of Virginia Militia and Maryland Militia 8 white men 12 free black men 1 freed slave 1 fugitive slave Casualties and
losses U.S.
</item 2>

<item 3>
John Brown's raid on Harpers Ferry
The militia forced the raiders inside the engine house . They broke into the guardroom and freed over two dozen prisoners . Eight militiamen were wounded .
Alburtis said that he could have ended the raid with help from other citizens . By 3 : 30 that afternoon , President James Buchanan ordered a company of U.S.
Marines ( the only government troops in the immediate area ) to march on Harpers Ferry under the command of Brevet Colonel Robert E. Lee , lieutenant colonel of
the 2nd U.S. Cavalry Regiment. Lee had been on leave from his regiment , stationed in Texas , when he was hastily called to lead the detachment and had to
command it while wearing his civilian clothes . October 18 ( edit ) This section needs additional citations for verification . Please help improve this article
by adding citations to reliable sources . Unsourced material may be challenged and removed . ( October 2017 ) ( Learn how and when to remove this template
message ) Lee first offered the role of attacking the engine house to the local militia units on the spot . Both militia commanders declined , and Lee turned to
the Marines .
</item 3>

<item 4>
John Brown's raid on Harpers Ferry
During the day , four townspeople were killed , including the mayor . Realizing his escape was cut , Brown took nine of his captives and moved into the smaller
engine house , which would come to be known as John Brown 's Fort . The raiders blocked entry of the windows and doors and traded sporadic gunfire with
surrounding forces . At one point Brown sent out his son , Watson , and Aaron Dwight Stevens with a white flag , but Watson was mortally wounded and Stevens was
shot and captured . The raid was rapidly failing . One of Brown 's men , William H. Leeman , panicked and made an attempt to flee by swimming across the Potomac
River , but he was shot and fatally injured while doing so . During the intermittent shooting , Brown 's other son , Oliver , was also hit ; he died after a
brief period . At around 3 : 00 p.m. a militia company led by Captain E.G. Alburtis arrived by train from Martinsburg , Virginia . Most of the militia members
were Baltimore & Ohio Railroad employees . The militia forced the raiders inside the engine house .
</item 4>

<item 5>
John Brown, an abolitionist, prepared at Maryland’s Kennedy Farm under an alias, recruiting a small, minimally trained interracial force and collecting Sharps
carbines and pikes. Despite warnings to officials and supporters, he led 22 men into Harpers Ferry on October 16, 1859, hoping to seize weapons, spark an
enslaved people’s uprising, and lead a campaign south. The uprising failed; Marines under Robert E. Lee stormed the raiders’ barricade, killing or capturing
</item 5>

<item 6>
John Brown's raid on Harpers Ferry
Regardless , one was killed trying to escape across the Potomac River , the other was wounded and later died in the Charles Town jail . ) 9 other civilians were
wounded . Marines Private Luke Quinn ( Killed during the storming of the engine house . ) Buried in Harpers Ferry Catholic Cemetery on Rte. 340 . Private
Matthew Ruppert ( Shot in the face during the storming of the engine house ; survived . ) Liberated slaves ( edit ) Mayor Beckham 's Will Book called for the
liberation of Isaac Gilbert , his wife , and three children upon his death . When Edwin Coppock killed Beckham the five slaves were free . See also ( edit )
List of incidents of civil unrest in the United States Notes ( edit ) Citations ( edit ) ^ Jump up to : `` John Brown 's raid on Harpers Ferry '' . History.com
. Retrieved 1 May 2015 . Jump up ^ In many books the town is called `` Harper 's Ferry . '' For example , `` Col.
</item 6>

<item 7>
John Brown's raid on Harpers Ferry
Marines : 1 killed 1 wounded Virginia and Maryland Militia : Unknown 10 killed 7 captured 5 escaped Civilians : 6 killed 9 wounded John Brown 's raid on Harper
's Ferry ( also known as John Brown 's raid or The raid on Harper 's Ferry ) was an effort by armed abolitionist John Brown to initiate an armed slave revolt in
1859 by taking over a United States arsenal at Harpers Ferry , Virginia . Brown 's party of 22 was defeated by a company of U.S. Marines , led by First
Lieutenant Israel Greene . Colonel Robert E. Lee was in overall command of the operation to retake the arsenal . John Brown had originally asked Harriet Tubman
and Frederick Douglass , both of whom he had met in his transformative years as an abolitionist in Springfield , Massachusetts , to join him in his raid , but
Tubman was prevented by illness and Douglass declined , as he believed Brown 's plan would fail .
</item 7>

<item 8>
John Brown's raid on Harpers Ferry
2007 On Sunday night , October 16 , 1859 , Brown left four of his men behind as a rear - guard : his son , Owen Brown , Barclay Coppock , and Frank Meriam ; he
led the rest into the town of Harpers Ferry , Virginia . Brown detached a party under John Cook Jr. to capture Colonel Lewis Washington , great - grandnephew of
George Washington , at his nearby Beall - Air estate , some of his slaves , and two relics of George Washington : a sword allegedly presented to Washington by
Frederick the Great and two pistols given by Marquis de Lafayette , which Brown considered talismans . The party carried out its mission and returned via the
Allstadt House , where they took more hostages . Brown 's main party captured several watchmen and townspeople in Harpers Ferry . Brown 's men needed to capture
the weapons and escape before word could be sent to Washington . The raid was going well for Brown 's men . They cut the telegraph wire and seized a Baltimore &
Ohio train passing through . A free black man was the first casualty of the raid .
</item 8>

<item 9>
John Brown's raid on Harpers Ferry
A free black man was the first casualty of the raid . Hayward Shepherd , an African - American baggage handler on the train , confronted the raiders ; they shot
and killed him . For some reason , Brown let the train continue , and the conductor alerted the authorities down the line . Brown had been sure that he would
win the support of local slaves in joining the rebellion , but a massive uprising did not occur , because word had not been spread about the uprising , so the
slaves nearby did not know about it . Although the white townspeople soon began to fight back against the raiders , Brown 's men succeeded in capturing the
armory that evening . October 17 ( edit ) Army workers discovered Brown 's men early on the morning of October 17 . Local militia , farmers and shopkeepers
surrounded the armory . When a company of militia captured the bridge across the Potomac River , any route of escape for the raiders was cut off . During the
day , four townspeople were killed , including the mayor .
</item 9>

<item 10>
In 1859, abolitionist John Brown led 22 men in a raid on the Harpers Ferry arsenal, hoping to spark a slave revolt and seize its estimated 100,000 weapons. U.S.
Marines, led by Lt. Israel Greene under Col. Robert E. Lee, defeated the raiders. Harriet Tubman could not join because of illness; Frederick Douglass refused,
predicting failure, while Shields Green joined. Reported losses included one Marine killed and one wounded, six civilians killed
</item 10>
=== END ITEM id=01f47b3891dd87d7 ===

=== ITEM id=5aa42d5c0def4f81 ===
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
Using perspective pinhole projections , a point on L projected on the image plane will have coordinates defined as , x ′ = f x / z = f d − bz / az y ′ = f y / z
= f c / z This is the parametric representation of the image L ′ of the line L with z as the parameter . When z → − ∞ it stops at the point ( x ′ , y ′ ) = ( −
fb / a , 0 ) on the x ′ axis of the image plane . This is the vanishing point corresponding to all parallel lines with slope − b / a in the plane π . All
vanishing points associated with different lines with different slopes belonging to plane π will lie on the x ′ axis , which in this case is the horizon line .
2 . Let A , B , and C be three mutually orthogonal straight lines in space and v ≡ ( x , y , f ) , v ≡ ( x , y , f ) , v ≡ ( x , y , f ) be the three
corresponding vanishing points respectively .
</item 1>

<item 2>
Vanishing point
With the assumption that the environment consists of objects with only parallel or perpendicular sides , also called Lego - land , using vanishing points
constructed in a single image of the scene they recovered the 3D geometry of the scene . Similar ideas are also used in the field of robotics , mainly in
navigation and autonomous vehicles , and in areas concerned with object detection . See also ( edit ) Graphical projection References ( edit ) ^ Jump up to :
Kirsti Andersen ( 2007 ) Geometry of an Art , p. xxx , Springer , ISBN 0 - 387 - 25961 - 9 ^ Jump up to : B. Caprile , V. Torre ( 1 ) `` Using Vanishing Points
for Camera Calibration '' , International Journal of Computer Vision , Volume 4 , Issue 2 , pp. 127 - 139 , March 1990 Jump up ^ H. Ditton ( 1712 ) Treatise on
Perspective , p 45 Jump up ^ S.T. Barnard ' Interpreting Perspective Images `` , Artificial Intelligence 21 , 1983 , pp.
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
Vanishing point
If we know the coordinates of one of these points , say v , and the direction of a straight line on the image plane , which passes through a second point , say
v , we can compute the coordinates of both v and v 3 . Let A , B , and C be three mutually orthogonal straight lines in space and v ≡ ( x , y , f ) , v ≡ ( x ,
y , f ) , v ≡ ( x , y , f ) be the three corresponding vanishing points respectively . The orthocenter of the triangle with vertices in the three vanishing
points is the intersection of the optical axis and the image plane . Curvilinear and reverse perspective ( edit ) A curvilinear perspective is a drawing with
either 4 or 5 vanishing points . In 5 - point perspective the vanishing points are mapped into a circle with 4 vanishing points at the cardinal headings N , W ,
S , E and one at the circle 's origin . A reverse perspective is a drawing with vanishing points that are placed outside the painting with the illusion that
they are `` in front of '' the painting . Single point perspective projection . Double point perspective projection .
</item 4>

<item 5>
Vanishing point
say D , will have the same vanishing point or converge at the same vanishing points . Mathematically , let q ≡ ( x , y , f ) be a point lying on the image plane
, where f is the focal length ( of the camera associated with the image ) , and let v ≡ ( x / h , y / h , f / h ) be the unit vector associated with q , where h
= √ x + y + f . If we consider a straight line in space S with the unit vector n ≡ ( n , n , n ) and its vanishing point v , the unit vector associated with v
is equal to n , assuming both are assumed to point towards the image plane . When the image plane is parallel to two world - coordinate axes , lines parallel to
the axis which is cut by this image plane will meet at infinity i.e. at the vanishing point . Lines parallel to the other two axes will not form vanishing
points as they are parallel to the image plane . This is one - point perspective .
</item 5>

<item 6>
Drawing
Typically this convergence point is somewhere along the horizon , as buildings are built level with the flat surface . When multiple structures are aligned with
each other , such as buildings along a street , the horizontal tops and bottoms of the structures typically converge at a vanishing point . When both the fronts
and sides of a building are drawn , then the parallel lines forming a side converge at a second point along the horizon ( which may be off the drawing paper . )
This is a two - point perspective . Converging the vertical lines to a third point above or below the horizon then produces a three - point perspective . Depth
can also be portrayed by several techniques in addition to the perspective approach above . Objects of similar size should appear ever smaller the further they
are from the viewer . Thus the back wheel of a cart appears slightly smaller than the front wheel . Depth can be portrayed through the use of texture . As the
texture of an object gets further away it becomes more compressed and busy , taking on an entirely different character than if it was close . Depth can also be
portrayed by reducing the contrast in more distant objects , and by making their colors less saturated .
</item 6>

<item 7>
Vanishing point
The horizon line is a theoretical line that represents the eye level of the observer . If the object is below the horizon line , its vanishing lines angle up to
the horizon line . If the object is above , they slope down . All vanishing lines end at the horizon line . Properties of vanishing points ( edit ) 1 .
Projections of two sets of parallel lines lying in some plane π appear to converge , i.e. the vanishing point associated with that pair , on a horizon line , or
vanishing line H formed by the intersection of the image plane with the plane parallel to π and passing through the pinhole . Proof : Consider the ground plane
π , as y = c which is , for the sake of simplicity , orthogonal to the image plane . Also , consider a line L that lies in the plane π , which is defined by the
equation ax + bz = d .
</item 7>

<item 8>
Vanishing point
Vanishing point - wikipedia Vanishing point Jump to : navigation , search For other uses , see Vanishing point ( disambiguation ) . A vanishing point can be
seen at the far end of this railroad . A vanishing point is a point on the image plane of a perspective drawing where the two - dimensional perspective
projections ( or drawings ) of mutually parallel lines in three - dimensional space appear to converge . When the set of parallel lines is perpendicular to a
picture plane , the construction is known as one - point perspective , and their vanishing point corresponds to the oculus , or `` eye point '' , from which the
image should be viewed for correct perspective geometry . Traditional linear drawings use objects with one to three sets of parallels , defining one to three
vanishing points . Part of a series on Graphical projection Planar ( show ) Parallel projection Orthographic projection Multiview projection Axonometric
projection Isometric projection Oblique projection Perspective projection Curvilinear perspective Reverse perspective Views ( show ) 2.
</item 8>

<item 9>
Vanishing point
5 D Bird 's - eye view Cross section Cutaway drawing Exploded view drawing Fisheye lens Panorama Worm 's - eye view Zoom lens Topics ( show ) 3D projection
Anamorphosis Axonometry Computer graphics Computer - aided design Descriptive geometry Engineering drawing Map projection Picture plane Plans ( drawings )
Projection ( linear algebra ) Projection plane Projective geometry Stereoscopy Technical drawing Vanishing point Video game graphics Viewing frustum Contents (
hide ) 1 Vector notation 2 Theorem 3 Vanishing line 4 Properties of vanishing points 5 Curvilinear and reverse perspective 6 Detection of vanishing points 7
Applications of vanishing points 8 See also 9 References 10 External links Vector notation ( edit ) A 2D construction of perspective viewing , showing the
formation of a vanishing point The vanishing point may also be referred to as the `` direction point '' , as lines having the same directional vector , say D ,
will have the same vanishing point or converge at the same vanishing points .
</item 9>

<item 10>
Vanishing point
She notes , in terms of projective geometry , the vanishing point is the image of the point at infinity associated with L , as the sightline from O through the
vanishing point is parallel to L . Vanishing line ( edit ) As a vanishing point originates in a line , so a vanishing line originates in a plane α that is not
parallel to the picture π . Given the eye point O , and β the plane parallel to α and lying on O , then the vanishing line of α is β ∩ π . For example , when α
is the ground plane and β is the horizon plane , then the vanishing line of α is the horizon line β ∩ π . Anderson notes , `` Only one particular vanishing line
occurs , often referred to as the `` horizon '' . To put it simply , the vanishing line of some plane , say α , is obtained by the intersection of the image
plane with another plane , say β , parallel to the plane of interest ( α ) , passing through the camera center . For different sets of lines parallel to this
plane α , their respective vanishing points will lie on this vanishing line .
</item 10>
=== END ITEM id=5aa42d5c0def4f81 ===
