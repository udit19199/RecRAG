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

=== ITEM id=0b7100aeeccf253e ===
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
My Next Guest Needs No Introduction with David Letterman
On January 5 , 2018 , Netflix released further information about the series . It was announced that Letterman 's guests in the first season would include Barack
Obama , George Clooney , Malala Yousafzai , Jay - Z , Tina Fey , and Howard Stern . It was also announced that the series would premiere on January 12 , 2018
with Letterman 's interview with Obama . This interview was Obama 's first since leaving office . Music ( edit ) The show 's theme song and bridging music were
written and recorded by Letterman 's long - time bandleader , Paul Shaffer . Soon after it was announced publicly that Netflix had given the series an order ,
Shaffer received a phone call from Letterman asking him to work on the show . Soon after , Shaffer began to receive cuts of episodes from the first season and
he started to put music in afterwards where the director thought it was needed . In developing the sound of the show 's music , Shaffer initially looked to
Letterman for guidance .
</item 1>

<item 2>
My Next Guest Needs No Introduction with David Letterman
`` I Had a Paper Route Too '' Jay - Z April 6 , 2018 ( 2018 - 04 - 06 ) Cold open : None On - location segment : Letterman visits Shangri La Studio , speaks
with music producer Rick Rubin and listens to Madison Ryann Ward recording . 5 `` TBA '' Tina Fey May 4 , 2018 6 `` TBA '' Howard Stern May 31 , 2018 Reception
( edit ) The first season of My Next Guest Needs No Introduction has been met with a positive response from critics . On the review aggregation website Rotten
Tomatoes , the first season holds a 80 % approval rating with an average rating of 6.4 out of 10 based on 25 reviews . Metacritic , which uses a weighted
average , assigned the season a score of 70 out of 100 based on 13 reviews , indicating `` generally favorable reviews '' . References ( edit ) Jump up ^
Andreeva , Nellie ( August 8 , 2017 ) . `` David Letterman Returns To Television With Netflix Series '' . Deadline .
</item 2>

<item 3>
My Next Guest Needs No Introduction with David Letterman
Magic School Bus Rides Again Suburra Mindhunter Super Monsters Cuando conocí al Chapo : La historia de Kate del Castillo Beyond Stranger Things Blazing Transfer
Student The Punisher Shot in the Dark Stretch Armstrong and the Flex Fighters She 's Gotta Have It Dark Wormwood The Toys That Made Us Since 2018 The End of the
F * * * ing World My Next Guest Needs No Introduction with David Letterman Trolls : The Beat Goes On ! Llama Llama The New Legends of Monkey Altered Carbon
Dirty Money Queer Eye First Team : Juventus The Joel McHale Show with Joel McHale Ugly Delicious B : The Beginning Flint Town Girls Incarcerated A.I.C.O. -
Incarnation - Nailed It ! On My Block Wild Wild Country Alexa & Katie The Mechanism Requiem Sword Gai : The Animation Lost Song Troy : Fall of a City The Boss
Baby : Back in Business Lost in Space Aggretsuko Spy Kids : Mission Critical The Letdown Luis Miguel Continuations Arrested Development ( season 4 , 2013 ) The
Killing ( season 4 , 2014 ) Star Wars : The Clone Wars ( season 6 ,
</item 3>

<item 4>
My Next Guest Needs No Introduction with David Letterman
Seuss ' Green Eggs and Ham ( 2018 ) Harvey Street Kids ( 2018 ) Hilda ( 2018 ) Insatiable ( 2018 ) Kingdom ( 2018 ) Kiss Me First ( 2018 ) Knights of the Zodiac
: Saint Seiya ( 2018 ) La Casa de las Flores ( 2018 ) Maniac ( 2018 ) The Ballad of Buster Scruggs ( 2018 ) The Epic Tales of Captain Underpants ( 2018 ) The
Umbrella Academy ( 2018 ) Carmen Sandiego ( 2019 ) Jinn ( 2019 ) Messiah ( 2019 ) Money Heist ( season 3 , 2019 ) Tales of the City ( 2019 ) Top Boy ( season 3
, 2019 ) The Last Kids on Earth ( 2019 ) Wizards ( 2019 ) Best Worst Weekend Ever ( TBA ) Chambers ( TBA ) Cursed ( TBA ) Dead to Me ( TBA ) Fast & Furious (
TBA ) Ghoul ( TBA ) Nightflyers ( TBA ) Norm Macdonald Has a Show ( TBA ) Paradise , P.
</item 4>

<item 5>
My Next Guest Needs No Introduction with David Letterman
2017 ) Glitch ( season 2 , 2017 ) Ended Lilyhammer ( 2012 -- 14 ) Hemlock Grove ( 2013 -- 15 ) Turbo FAST ( 2013 -- 16 ) VeggieTales in the House ( 2014 -- 16 )
Marco Polo ( 2014 -- 16 ) All Hail King Julien ( 2014 -- 17 ) The Adventures of Puss in Boots ( 2015 -- 18 ) Richie Rich ( 2015 ) Bloodline ( 2015 -- 17 )
Between ( 2015 -- 16 ) Wet Hot American Summer : First Day of Camp ( 2015 ) Popples ( 2015 -- 16 ) The Mr. Peabody & Sherman Show ( 2015 -- 17 ) Care Bears and
Cousins ( 2015 -- 16 ) W / Bob & David ( 2015 ) Atelier ( 2015 ) Dawn of the Croods ( 2015 -- 17 ) Chelsea Does ( 2016 ) Love ( 2016 -- 18 ) Lego Bionicle : The
Journey to One ( 2016 ) Netflix Presents :
</item 5>

<item 6>
My Next Guest Needs No Introduction with David Letterman
Rotten Tomatoes . Retrieved January 13 , 2018 . Jump up ^ `` My Next Guest Needs No Introduction With David Letterman : Season 1 - Metacritic '' . Metacritic .
Retrieved January 13 , 2018 . External links ( edit ) Official website David Letterman Television hosting The David Letterman Show ( 1980 ) Late Night with
David Letterman ( 1982 -- 1993 ) Late Show with David Letterman ( 1993 -- 2015 ) My Next Guest Needs No Introduction with David Letterman ( 2018 ) Companies The
Letterman Foundation for Courtesy and Grooming Rahal Letterman Lanigan Racing Worldwide Pants Related Dorothy Mengering Netflix original series Current Since
2013 House of Cards Orange Is the New Black Since 2014 BoJack Horseman Since 2015 Unbreakable Kimmy Schmidt Chef 's Table Daredevil Grace and Frankie Residue
Sense8 Club de Cuervos Project Mc Dinotrux Narcos Master of None Jessica Jones F Is for Family Making a Murderer Real Rob Since 2016 Degrassi : Next Class
Fuller House Lego Friends : The Power of Friendship Flaked
</item 6>

<item 7>
My Next Guest Needs No Introduction with David Letterman
Washington Post . Retrieved January 13 , 2018 . Jump up ^ Leight , Elias ( February 2 , 2018 ) . `` Paul Shaffer on Writing Letterman 's Netflix Theme Song &
the Value of Humor in Pop '' . Billboard . Retrieved February 9 , 2018 . Jump up ^ Carr , Mary Kate ( April 24 , 2018 ) . `` What 's on Netflix in May : The
best shows and movies to watch this month '' . Entertainment Weekly . Retrieved April 24 , 2018 . Jump up ^ Douglas , Nick ( April 24 , 2018 ) . `` Here 's What
's Coming And Going From Netflix In May 2018 '' . Lifehacker . Retrieved April 24 , 2018 . Jump up ^ `` My Next Guest Needs No Introduction With David Letterman
: Season 1 - Rotten Tomatoes '' . Rotten Tomatoes . Retrieved January 13 , 2018 .
</item 7>

<item 8>
My Next Guest Needs No Introduction with David Letterman
wikipedia.org/w/index.php?title=My_Next_Guest_Needs_No_Introduction_with_David_Letterman&oldid=838074810 '' Categories : 2010s American comedy television series
2018 American television series debuts David Letterman English - language television programs Netflix original programming Hidden categories : Pages using
infobox television with editor parameter Episode lists with unformatted air dates Talk Contents About Wikipedia Italiano Nederlands Português Edit links This
page was last edited on 24 April 2018 , at 19 : 28 . About Wikipedia
</item 8>

<item 9>
My Next Guest Needs No Introduction with David Letterman
My Next Guest Needs No Introduction with David Letterman - wikipedia My Next Guest Needs No Introduction with David Letterman Jump to : navigation , search My
Next Guest Needs No Introduction with David Letterman Genre Talk show Directed by Michael Bonfiglio Presented by David Letterman Composer ( s ) Paul Shaffer
Country of origin United States Original language ( s ) English No. of seasons No. of episodes 4 ( list of episodes ) Production Executive producer ( s ) Justin
Wilkes Dave Sirulnick Jon Kamen Tom Keany Producer ( s ) Mary Barclay Cinematography Declan Quinn Editor ( s ) Jesse Gordon Edward A. Bishop Paul Little Running
time 50 -- 56 minutes Production company ( s ) RadicalMedia Worldwide Pants Inc . Release Original network Netflix Original release January 12 , 2018 ( 2018 -
01 - 12 ) -- present External links Official website My Next Guest Needs No Introduction with David Letterman is an American web television talk show hosted by
David Letterman . The series consists of interviews with one guest per episode both inside and outside a studio setting .
</item 9>

<item 10>
My Next Guest Needs No Introduction with David Letterman
The series consists of interviews with one guest per episode both inside and outside a studio setting . The first season will consist of six episodes and will
include interviews with Barack Obama , George Clooney , Malala Yousafzai , Jay - Z , Tina Fey , and Howard Stern . Obama was the first guest on the show , with
the first episode released on Netflix on January 12 , 2018 . Contents ( hide ) 1 Production 1.1 Development 1.2 Music 2 Episodes 3 Reception 4 References 5
External links Production ( edit ) Development ( edit ) On August 8 , 2017 , it was announced that Netflix had ordered a six - episode interview series starring
David Letterman . Each hour - long episode will feature Letterman conducting a longform conversation with a single guest and in - the - field segments , in
which he will explore topics on his own . On January 5 , 2018 , Netflix released further information about the series .
</item 10>
=== END ITEM id=0b7100aeeccf253e ===

=== ITEM id=e8a7bb2255bb8c1f ===
Gold sentences and facts:
Sentence 0: Club Owner ( s ) Estimated combined net worth Source of wealth Arsenal ( more information ) E. Stanley Kroenke ( 67.05 % ) Alisher Usmanov ( 30.04 % ) 15 ! $22.4 B Kroenke Sports Enterprises Metal and mining , investment Bournemouth Maxim Demin ( 75 % ) PEAK6 Investments ( 25 % ) Petrochemicals trading Seven Developments Brighton & Hove Albion Tony Bloom ( 75.61 % ) 12 ! $1.7 B Online Gambling , Real Estate , Property , Land Development & Investments Burnley Mike Garlick ( 49.3 % ) John Banaszkiewicz ( 27.55 % ) 18 ! $80 M -- - Michael Bailey Associates Freight Investor Services Chelsea ( more information ) Roman Abramovich 03 ! $9.3 B Oil and industry Crystal Palace Steve Parish Martin Long Joshua Harris David S. Blitzer Jeremy Hosking Stephen Browett 15 ! $65 M $100 M $3.2 B $1.3 B $250 M $110 M Tag Worldwide Churchill Insurance Apollo Global Management Blackstone Group Marathon Asset Management Farr Vintners Everton ( more information ) Farhad Moshiri ( 49.9 % ) Bill Kenwright CBE Jon Woods 15 ! $2.3 B Steel and energy Theatre production Planet Hollywood Ocean Software Huddersfield Town Dean Hoyle 09 ! $475 M Card Factory Leicester City The Srivaddhanaprabha Family 10 ! $4.9 B King Power International Group Liverpool ( more information ) John W. Henry 11 ! $2.5 B Fenway Sports Group Manchester City ( more information ) Mansour bin Zayed Al Nahyan ( 87 % ) China Media Capital ( 13 % ) 01 ! $44 B Oil and industry investment Government Manchester United ( more information ) The Glazer Family 06 ! $4.9 B First Allied Corporation , Tampa Bay Buccaneers Newcastle United ( more information ) Mike Ashley 04 ! $3.3 B Sports Direct International Southampton Gao Jisheng ( 80 % ) Katharina Liebherr ( 20 % ) 07 ! $3.9 B + Lander Sports Development Co Ltd Inheritance Stoke City The Coates Family 08 ! $3.1 B Bet365 Swansea City Stephen Kaplan & Jason Levien ( 68 % ) Swansea City Supporters Trust ( 21.1 % ) -- - -- - Memphis Grizzlies and D.C. United Investment trust Tottenham Hotspur ( more information ) Joe Lewis ( 70.6 % of 85 % holding company ) Daniel Levy ( 29.4 % of 85 % holding company ) 05 ! $5.6 B -- - Tavistock Group ENIC Group Watford Gino Pozzo -- - Businessman West Bromwich Albion ( more information ) Guochuan Lai ( 87.8 % ) 18 ! $6.2 B Investment West Ham United ( more information ) David Sullivan ( 51.1 % ) David Gold ( 35.1 % ) Straumur Investment Bank ( 10 % ) 13 ! $520 M $325 M -- - Daily Sport , Sunday Sport Gold Group International Straumur Investment Bank
  Fact 0.0: Arsenal's owner E. Stanley Kroenke holds 67.05% of the club.
  Fact 0.1: Arsenal's owner Alisher Usmanov holds 30.04% of the club.
  Fact 0.2: Arsenal's estimated combined owner net worth is $22.4 B.
  Fact 0.3: Arsenal's owners' source of wealth is metal and mining and investment.
  Fact 0.4: Kroenke Sports Enterprises is listed for Arsenal.
  Fact 0.5: Bournemouth's owner Maxim Demin holds 75% of the club.
  Fact 0.6: Bournemouth's owner PEAK6 Investments holds 25% of the club.
  Fact 0.7: Bournemouth's source of wealth is petrochemicals trading.
  Fact 0.8: Seven Developments is listed for Bournemouth.
  Fact 0.9: Brighton & Hove Albion's owner Tony Bloom holds 75.61% of the club.
  Fact 0.10: Brighton & Hove Albion's owner net worth is $1.7 B.
  Fact 0.11: Brighton & Hove Albion's source of wealth is online gambling, real estate, property, land development and investments.
  Fact 0.12: Burnley's owner Mike Garlick holds 49.3% of the club.
  Fact 0.13: Burnley's owner John Banaszkiewicz holds 27.55% of the club.
  Fact 0.14: Burnley's estimated combined owner net worth is $80 M.
  Fact 0.15: Burnley's sources of wealth are Michael Bailey Associates and Freight Investor Services.
  Fact 0.16: Chelsea's owner is Roman Abramovich.
  Fact 0.17: Chelsea's owner net worth is $9.3 B.
  Fact 0.18: Chelsea's source of wealth is oil and industry.
  Fact 0.19: Crystal Palace's owners are Steve Parish, Martin Long, Joshua Harris, David S. Blitzer, Jeremy Hosking and Stephen Browett.
  Fact 0.20: Crystal Palace's owners' net worths are listed as $65 M, $100 M, $3.2 B, $1.3 B, $250 M and $110 M.
  Fact 0.21: Crystal Palace's sources of wealth are listed as Tag Worldwide, Churchill Insurance, Apollo Global Management, Blackstone Group, Marathon Asset Management and Farr Vintners.
  Fact 0.22: Everton's owner Farhad Moshiri holds 49.9% of the club.
  Fact 0.23: Everton's owners include Bill Kenwright CBE and Jon Woods.
  Fact 0.24: Everton's estimated combined owner net worth is $2.3 B.
  Fact 0.25: Everton's sources of wealth are steel and energy, theatre production, Planet Hollywood and Ocean Software.
  Fact 0.26: Huddersfield Town's owner is Dean Hoyle.
  Fact 0.27: Huddersfield Town's owner net worth is $475 M.
  Fact 0.28: Huddersfield Town's source of wealth is Card Factory.
  Fact 0.29: Leicester City's owner is The Srivaddhanaprabha Family.
  Fact 0.30: Leicester City's owner net worth is $4.9 B.
  Fact 0.31: Leicester City's source of wealth is King Power International Group.
  Fact 0.32: Liverpool's owner is John W. Henry.
  Fact 0.33: Liverpool's owner net worth is $2.5 B.
  Fact 0.34: Liverpool's source of wealth is Fenway Sports Group.
  Fact 0.35: Manchester City's owner Mansour bin Zayed Al Nahyan holds 87% of the club.
  Fact 0.36: Manchester City's owner China Media Capital holds 13% of the club.
  Fact 0.37: Manchester City's owner net worth is $44 B.
  Fact 0.38: Manchester City's sources of wealth are oil and industry investment and government.
  Fact 0.39: Manchester United's owner is The Glazer Family.
  Fact 0.40: Manchester United's owner net worth is $4.9 B.
  Fact 0.41: Manchester United's sources of wealth are First Allied Corporation and Tampa Bay Buccaneers.
  Fact 0.42: Newcastle United's owner is Mike Ashley.
  Fact 0.43: Newcastle United's owner net worth is $3.3 B.
  Fact 0.44: Newcastle United's source of wealth is Sports Direct International.
  Fact 0.45: Southampton's owner Gao Jisheng holds 80% of the club.
  Fact 0.46: Southampton's owner Katharina Liebherr holds 20% of the club.
  Fact 0.47: Southampton's owner net worth is $3.9 B.
  Fact 0.48: Southampton's source of wealth is Lander Sports Development Co Ltd and inheritance.
  Fact 0.49: Stoke City's owner is The Coates Family.
  Fact 0.50: Stoke City's owner net worth is $3.1 B.
  Fact 0.51: Stoke City's source of wealth is Bet365.
  Fact 0.52: Swansea City's owners Stephen Kaplan and Jason Levien hold 68% of the club.
  Fact 0.53: Swansea City's Supporters Trust holds 21.1% of the club.
  Fact 0.54: Swansea City's sources of wealth are Memphis Grizzlies, D.C. United and investment trust.
  Fact 0.55: Tottenham Hotspur's owner Joe Lewis holds 70.6% of the 85% holding company.
  Fact 0.56: Tottenham Hotspur's owner Daniel Levy holds 29.4% of the 85% holding company.
  Fact 0.57: Tottenham Hotspur's owner net worth is $5.6 B.
  Fact 0.58: Tottenham Hotspur's sources of wealth are Tavistock Group and ENIC Group.
  Fact 0.59: Watford's owner is Gino Pozzo.
  Fact 0.60: Watford's owner is described as a businessman.
  Fact 0.61: West Bromwich Albion's owner Guochuan Lai holds 87.8% of the club.
  Fact 0.62: West Bromwich Albion's owner net worth is $6.2 B.
  Fact 0.63: West Bromwich Albion's source of wealth is investment.
  Fact 0.64: West Ham United's owner David Sullivan holds 51.1% of the club.
  Fact 0.65: West Ham United's owner David Gold holds 35.1% of the club.
  Fact 0.66: West Ham United's owner Straumur Investment Bank holds 10% of the club.
  Fact 0.67: West Ham United's owners' net worths are listed as $520 M and $325 M.
  Fact 0.68: West Ham United's sources of wealth are Daily Sport, Sunday Sport, Gold Group International and Straumur Investment Bank.
Sentence 1: Club Owner ( s ) Estimated combined net worth ( £ m ) Source of wealth Aston Villa ( more information ) Dr. Tony Xia $1.2 B ! $1.2 B Recon Group Barnsley Patrick Cryne 14 ! $40 M iSOFT Birmingham City ( more information ) Paul Suen ( 60.78 % ) Wang Lei ( 5.28 % ) Carson Yeung ( 6.51 % ) 21 ! $570 M Birmingham International Holdings Bolton Wanderers Eddie Davies CBE 13 ! $52 M Strix Brentford Matthew Benham ( 96 % ) Bristol City Stephen Lansdown 01 ! $1.9 B Hargreaves Lansdown Burton Albion Ben Robinson Cardiff City Vincent Tan ( 51 % ) 01 ! $820 M Berjaya Group Derby County Mel Morris $520 M King Fulham ( more information ) Shahid Khan 09 ! $8.9 B Flex - N-Gate Corp ( automobile parts manufacturer ) Hull City The Allam Family 15 ! $325 M Allam Marine Ipswich Town Marcus Evans 02 ! $825 M Marcus Evans Group Leeds United Andrea Radrizzani Eleonora Sports Ltd Aser Group Company Middlesbrough Steve Gibson 10 ! $260 M Bulkhaul Limited Millwall John Berylson Richard Smith 11 ! $175 M Chestnut Hill Ventures Norwich City ( more information ) Delia Smith and Michael Wynn - Jones ( 53 % ) Michael Foulger ( 15 % ) 20 ! $30 M Food industry Publishing Poultry Nottingham Forest Evangelos Marinakis 21 ! $650 M Owner of Olympiacos F.C. Preston North End Trevor Hemmings CVO ( 51.4 % ) 02 ! $720 M Blackpool Tower , horse racing , investment , Northern Trust , Sportech Queens Park Rangers ( more information ) Tony Fernandes ( 66 % ) Lakshmi Mittal ( 33 % ) 03 ! $345 M $16.4 B Air Asia , Tune Group ArcelorMittal Reading ( more information ) Narin Niruttinanon ( 50 % ) Sasima Srivikorn ( 25 % ) Sumrith Thanakarnjanasuth ( 25 % ) Sheffield United Abdullah bin Musa'ed ( 50 % ) 03 ! $155 M Property Oil Sheffield Wednesday Dejphon Chansiri Thai Union Group Sunderland Ellis Short 09 ! $2.9 B Lone Star Funds , Skibo Castle Wolverhampton Wanderers Fosun International ( parent company ) Guo Guangchang ( co-owner of Fosun ) $6.2 B Investment
  Fact 1.0: Aston Villa's owner is Dr. Tony Xia.
  Fact 1.1: Aston Villa's owner net worth is $1.2 B.
  Fact 1.2: Aston Villa's source of wealth is Recon Group.
  Fact 1.3: Barnsley's owner is Patrick Cryne.
  Fact 1.4: Barnsley's owner net worth is $40 M.
  Fact 1.5: Barnsley's source of wealth is iSOFT.
  Fact 1.6: Birmingham City's owners are Paul Suen (60.78%), Wang Lei (5.28%) and Carson Yeung (6.51%).
  Fact 1.7: Birmingham City's estimated combined owner net worth is $570 M.
  Fact 1.8: Birmingham City's source of wealth is Birmingham International Holdings.
  Fact 1.9: Bolton Wanderers' owner is Eddie Davies CBE.
  Fact 1.10: Bolton Wanderers' owner net worth is $52 M.
  Fact 1.11: Bolton Wanderers' source of wealth is Strix.
  Fact 1.12: Brentford's owner Matthew Benham holds 96% of the club.
  Fact 1.13: Bristol City's owner is Stephen Lansdown.
  Fact 1.14: Bristol City's owner net worth is $1.9 B.
  Fact 1.15: Bristol City's source of wealth is Hargreaves Lansdown.
  Fact 1.16: Burton Albion's owner is Ben Robinson.
  Fact 1.17: Cardiff City's owner Vincent Tan holds 51% of the club.
  Fact 1.18: Cardiff City's owner net worth is $820 M.
  Fact 1.19: Cardiff City's source of wealth is Berjaya Group.
  Fact 1.20: Derby County's owner is Mel Morris.
  Fact 1.21: Derby County's owner net worth is $520 M.
  Fact 1.22: Fulham's owner is Shahid Khan.
  Fact 1.23: Fulham's owner net worth is $8.9 B.
  Fact 1.24: Fulham's source of wealth is Flex-N-Gate Corp, an automobile parts manufacturer.
  Fact 1.25: Hull City's owner is The Allam Family.
  Fact 1.26: Hull City's owner net worth is $325 M.
  Fact 1.27: Hull City's source of wealth is Allam Marine.
  Fact 1.28: Ipswich Town's owner is Marcus Evans.
  Fact 1.29: Ipswich Town's owner net worth is $825 M.
  Fact 1.30: Ipswich Town's source of wealth is Marcus Evans Group.
  Fact 1.31: Leeds United's owners are Andrea Radrizzani, Eleonora Sports Ltd and Aser Group Company.
  Fact 1.32: Middlesbrough's owner is Steve Gibson.
  Fact 1.33: Middlesbrough's owner net worth is $260 M.
  Fact 1.34: Middlesbrough's source of wealth is Bulkhaul Limited.
  Fact 1.35: Millwall's owners are John Berylson and Richard Smith.
  Fact 1.36: Millwall's owner net worth is $175 M.
  Fact 1.37: Millwall's source of wealth is Chestnut Hill Ventures.
  Fact 1.38: Norwich City's owners Delia Smith and Michael Wynn-Jones hold 53% of the club.
  Fact 1.39: Norwich City's owner Michael Foulger holds 15% of the club.
  Fact 1.40: Norwich City's owner net worth is $30 M.
  Fact 1.41: Norwich City's sources of wealth are food industry, publishing and poultry.
  Fact 1.42: Nottingham Forest's owner is Evangelos Marinakis.
  Fact 1.43: Nottingham Forest's owner net worth is $650 M.
  Fact 1.44: Nottingham Forest's owner is also owner of Olympiacos F.C.
  Fact 1.45: Preston North End's owner Trevor Hemmings CVO holds 51.4% of the club.
  Fact 1.46: Preston North End's owner net worth is $720 M.
  Fact 1.47: Preston North End's sources of wealth are Blackpool Tower, horse racing, investment, Northern Trust and Sportech.
  Fact 1.48: Queens Park Rangers' owner Tony Fernandes holds 66% of the club.
  Fact 1.49: Queens Park Rangers' owner Lakshmi Mittal holds 33% of the club.
  Fact 1.50: Queens Park Rangers' owners' net worths are listed as $345 M and $16.4 B.
  Fact 1.51: Queens Park Rangers' sources of wealth are Air Asia, Tune Group and ArcelorMittal.
  Fact 1.52: Reading's owners are Narin Niruttinanon (50%), Sasima Srivikorn (25%) and Sumrith Thanakarnjanasuth (25%).
  Fact 1.53: Sheffield United's owner Abdullah bin Musa'ed holds 50% of the club.
  Fact 1.54: Sheffield United's owner net worth is $155 M.
  Fact 1.55: Sheffield United's sources of wealth are property and oil.
  Fact 1.56: Sheffield Wednesday's owner is Dejphon Chansiri.
  Fact 1.57: Sheffield Wednesday's source of wealth is Thai Union Group.
  Fact 1.58: Sunderland's owner is Ellis Short.
  Fact 1.59: Sunderland's owner net worth is $2.9 B.
  Fact 1.60: Sunderland's sources of wealth are Lone Star Funds and Skibo Castle.
  Fact 1.61: Wolverhampton Wanderers' owners are Fosun International (parent company) and Guo Guangchang (co-owner of Fosun).
  Fact 1.62: Wolverhampton Wanderers' owner net worth is $6.2 B.
  Fact 1.63: Wolverhampton Wanderers' source of wealth is investment.

Retrieved items:
<item 1>
List of owners of English football clubs
$260 M Bulkhaul Limited Millwall John Berylson Richard Smith 11 ! $175 M Chestnut Hill Ventures Norwich City ( more information ) Delia Smith and Michael Wynn -
Jones ( 53 % ) Michael Foulger ( 15 % ) 20 ! $30 M Food industry Publishing Poultry Nottingham Forest Evangelos Marinakis 21 ! $650 M Owner of Olympiacos F.C.
Preston North End Trevor Hemmings CVO ( 51.4 % ) 02 ! $720 M Blackpool Tower , horse racing , investment , Northern Trust , Sportech Queens Park Rangers ( more
information ) Tony Fernandes ( 66 % ) Lakshmi Mittal ( 33 % ) 03 ! $345 M $16.4 B Air Asia , Tune Group ArcelorMittal Reading ( more information ) Narin
Niruttinanon ( 50 % ) Sasima Srivikorn ( 25 % ) Sumrith Thanakarnjanasuth ( 25 % ) Sheffield United Abdullah bin Musa'ed ( 50 % ) 03 !
</item 1>

<item 2>
List of owners of English football clubs
$3.1 B Bet365 Swansea City Stephen Kaplan & Jason Levien ( 68 % ) Swansea City Supporters Trust ( 21.1 % ) -- - -- - Memphis Grizzlies and D.C. United
Investment trust Tottenham Hotspur ( more information ) Joe Lewis ( 70.6 % of 85 % holding company ) Daniel Levy ( 29.4 % of 85 % holding company ) 05 ! $5.6 B
-- - Tavistock Group ENIC Group Watford Gino Pozzo -- - Businessman West Bromwich Albion ( more information ) Guochuan Lai ( 87.8 % ) 18 ! $6.2 B Investment
West Ham United ( more information ) David Sullivan ( 51.1 % ) David Gold ( 35.1 % ) Straumur Investment Bank ( 10 % ) 13 ! $520 M $325 M -- - Daily Sport ,
Sunday Sport Gold Group International Straumur Investment Bank Football League Championship ( edit ) Club Owner ( s ) Estimated combined net worth ( £ m )
Source of wealth Aston Villa ( more information ) Dr. Tony Xia $1.2 B !
</item 2>

<item 3>
List of owners of English football clubs
Jump up ^ `` Shahid Khan '' . Forbes . Forbes.com LLC. 19 May 2017 . Retrieved 19 May 2017 . Jump up ^ `` Who 's Who At The KC '' . Hull City F.C. 25 June 2012
. Retrieved 9 May 2014 . Jump up ^ `` Sunday Times Rich List 2017 reveals Yorkshire 's 20 wealthiest people '' . Hull Daily Mail . Archived from the original on
2 May 2017 . Retrieved 19 May 2017 . Jump up ^ www.calcioefinanza.it Leeds http://www.calcioefinanza.it/2017/01/04/leeds-cellino-radrizzani/ Leeds Check url =
value ( help ) . Retrieved 2017 - 01 - 10 . Missing or empty title = ( help ) Jump up ^ leeds united
https://www.leedsunited.com/news/club-news/21921/club-statement . Retrieved 2017 - 05 - 23 .
</item 3>

<item 4>
List of owners of English football clubs
A.F.C. Sunderland . Retrieved 9 May 2014 . Jump up ^ `` How the English Premier League Clubs ' Owners Got Rich '' . Archived from the original on 13 January
2015 . Retrieved 11 December 2016 . Jump up ^ `` # 226 Guo Guangchang '' . Forbes . Retrieved 19 May 2017 . Jump up ^ `` Bury FC Ownership '' . Bury F.C. 25
June 2012 . Retrieved 12 February 2013 . Jump up ^ `` Directors '' . Charlton Athletic F.C. 30 July 2013 . Retrieved 12 May 2014 . Jump up ^ `` Who 's Who at
Doncaster Rovers '' . Doncaster Rovers F.C . Retrieved 12 February 2013 . Jump up ^ `` Terry Bramall Awarded CBE '' . Doncaster Rovers F.C. 29 December 2012 .
Retrieved 13 February 2013 . Jump up ^ `` Ownership '' . Fleetwood Town F.C .
</item 4>

<item 5>
List of owners of English football clubs
$250 M DW Sports Fitness , JJB Sports League Two ( edit ) Club Owner ( s ) Estimated combined net worth Source of wealth Accrington Stanley The Styring Family (
12 % ) Accrington Stanley Supporters Fund ( 12 % ) Peter Marsden ( 10 % ) Investment trust Barnet Blackpool Owen Oyston ( 80 % ) Valeri Belokon ( 25 % ) 07 !
$105 M $195 M Publishing , Media & Estate Agency Finance , Media , Food Industry & Mining Cambridge United Grosvenor Fund Management Grosvenor Group Owned by
the Duke of Westminster and family Property Carlisle United ( more information ) A. Jenkins ( 37 % of 93 % holding company ) J. Nixon ( 18 % ) S. Pattison ( 18
% ) The United Trust ( 25.4 % ) Pioneer Foods Businessman ( retired ) Carlisle Glass Investment Trust Cheltenham Town Paul Baker Chesterfield Dave Allen
Colchester United Robbie Cowling 09 !
</item 5>

<item 6>
List of owners of English football clubs
List of owners of English football clubs - wikipedia List of owners of English football clubs Jump to : navigation , search This is a list of the current owners
of English football clubs , as well as ( in some cases ) their estimated net worth and source of wealth . All figures are listed in millions rather than
billions . For example , £ 1,000 m would be one billion ( 1,000,000,000 ) pounds . Only shareholders with a significant interest ( above 10 % ) are listed .
Contents ( hide ) 1 Premier League 2 Football League Championship 3 League One 4 League Two 5 References Premier League ( edit ) This list is incomplete ; you
can help by expanding it . Club Owner ( s ) Estimated combined net worth Source of wealth Arsenal ( more information ) E. Stanley Kroenke ( 67.05 % ) Alisher
Usmanov ( 30.04 % ) 15 !
</item 6>

<item 7>
List of owners of English football clubs
^ Jump up to : `` The Football Rich List 2011 / 12 : The Full 100 '' . Four Four Two . Retrieved 13 February 2013 . Jump up ^ `` Board of Directors / CEO '' .
Burnley F.C . Retrieved 9 May 2014 . Jump up ^ `` The Rich List 2014 '' . Hertfordshire Life . 31 March 2014 . Retrieved 12 May 2014 . Jump up ^ `` Club
Personnel '' . Chelsea F.C . Retrieved 9 August 2015 . Jump up ^ `` # 139 Roman Abramovich '' . Forbes . Retrieved 19 May 2017 . Jump up ^ `` # 745 Joshua
Harris '' . Forbes . Retrieved 19 May 2017 . Jump up ^ `` Senior changes at Tag Worldwide '' . Williams Lea. 24 October 2012 . Retrieved 16 February 2013 . Jump
up ^ `` # 896 Farhad Moshiri '' . Forbes . Retrieved 19 May 2017 .
</item 7>

<item 8>
List of owners of English football clubs
$155 M Property Oil Sheffield Wednesday Dejphon Chansiri Thai Union Group Sunderland Ellis Short 09 ! $2.9 B Lone Star Funds , Skibo Castle Wolverhampton
Wanderers Fosun International ( parent company ) Guo Guangchang ( co-owner of Fosun ) $6.2 B Investment League One ( edit ) Club Owner ( s ) Estimated combined
net worth Source of wealth AFC Wimbledon Wimbledon Football Club Supporters ' Society ( The Dons Trust ) Supporters Trust Blackburn Rovers Balaji & Venkatesh
Rao ( 99.9 % ) 04 ! $650 M Venkateshwara Hatcheries Group Bradford City Bristol Rovers Dwane Sports ( 92 % ) Banking and tourism Bury Save Our Shakers Trust (
63.8 % ) The Bury F.C. Supporters Society Ltd ( Forever Bury ) ( 11 % ) Investment trusts Charlton Athletic Roland Duchatelet ( 95 % ) 24 ! Staprix NV Doncaster
Rovers John Ryan Dick Watson ( 26 % ) Terry Bramall CBE 07 !
</item 8>

<item 9>
List of owners of English football clubs
Tony Xia $1.2 B ! $1.2 B Recon Group Barnsley Patrick Cryne 14 ! $40 M iSOFT Birmingham City ( more information ) Paul Suen ( 60.78 % ) Wang Lei ( 5.28 % )
Carson Yeung ( 6.51 % ) 21 ! $570 M Birmingham International Holdings Bolton Wanderers Eddie Davies CBE 13 ! $52 M Strix Brentford Matthew Benham ( 96 % )
Bristol City Stephen Lansdown 01 ! $1.9 B Hargreaves Lansdown Burton Albion Ben Robinson Cardiff City Vincent Tan ( 51 % ) 01 ! $820 M Berjaya Group Derby
County Mel Morris $520 M King Fulham ( more information ) Shahid Khan 09 ! $8.9 B Flex - N-Gate Corp ( automobile parts manufacturer ) Hull City The Allam
Family 15 ! $325 M Allam Marine Ipswich Town Marcus Evans 02 ! $825 M Marcus Evans Group Leeds United Andrea Radrizzani Eleonora Sports Ltd Aser Group Company
Middlesbrough Steve Gibson 10 !
</item 9>

<item 10>
List of owners of English football clubs
`` Trevor Hemmings ' Northern Trust returns to profit '' . The Daily Telegraph . London . Retrieved 12 February 2013 . Jump up ^ `` Company details '' . Queens
Park Rangers F.C. 10 July 2012 . Archived from the original on 10 August 2012 . Retrieved 12 February 2012 . Jump up ^ `` # 28 Anthony Fernandes '' . Forbes .
Retrieved 27 May 2014 . Jump up ^ `` # 56 Lakshmi Mittal '' . Forbes . Retrieved 19 May 2017 . Jump up ^ Details , Company . `` COMPANY DETAILS '' . Retrieved
29 July 2016 . Jump up ^ `` Fantasy football nut Saudi Prince Abdullah is joint owner of League One side Sheffield United '' . Daily Mail . London . 3 September
2013 . Retrieved 5 September 2013 . Jump up ^ `` Board & Management '' . A.F.C. Sunderland . Retrieved 9 May 2014 .
</item 10>
=== END ITEM id=e8a7bb2255bb8c1f ===

=== ITEM id=349099f4b2626778 ===
Gold sentences and facts:
Sentence 0: The current phrasing appears to have originated in the 1910 book Physics by Charles Riborg Mann and George Ransom Twiss . The question `` When a tree falls in a lonely forest , and no animal is near by to hear it , does it make a sound ? Why ? '' is posed along with many other questions to quiz readers on the contents of the chapter , and as such , is posed from a purely physical point of view .
  Fact 0.0: The current phrasing of the tree-falling question appears to have originated in the 1910 book Physics.
  Fact 0.1: The 1910 book Physics was written by Charles Riborg Mann and George Ransom Twiss.
  Fact 0.2: The question "When a tree falls in a lonely forest, and no animal is near by to hear it, does it make a sound? Why?" is posed along with many other questions to quiz readers on the contents of the chapter.
  Fact 0.3: The tree-falling question is posed from a purely physical point of view.
Sentence 1: Philosopher George Berkeley , in his work , A Treatise Concerning the Principles of Human Knowledge ( 1710 ) , proposes , `` But , say you , surely there is nothing easier than for me to imagine trees , for instance , in a park ( ... ) and nobody by to perceive them . ( ... ) The objects of sense exist only when they are perceived ; the trees therefore are in the garden ( ... ) no longer than while there is somebody by to perceive them . '' ( It is worth noting that the quote from section 45 is arguably a statement of an objection to Berkeley 's view , and not a proclamation of it . ) Nevertheless , Berkeley never actually wrote about the question .
  Fact 1.0: Philosopher George Berkeley wrote A Treatise Concerning the Principles of Human Knowledge, published in 1710.
  Fact 1.1: In A Treatise Concerning the Principles of Human Knowledge, Berkeley proposes a passage that begins "But, say you, surely there is nothing easier than for me to imagine trees, for instance, in a park ... and nobody by to perceive them."
  Fact 1.2: The passage quoted from Berkeley's treatise continues "The objects of sense exist only when they are perceived; the trees therefore are in the garden ... no longer than while there is somebody by to perceive them."
  Fact 1.3: The quote from section 45 of Berkeley's treatise is arguably a statement of an objection to Berkeley's view, not a proclamation of it.
  Fact 1.4: Berkeley never actually wrote about the tree-falling question.

Retrieved items:
<item 1>
If a tree falls in a forest
-- e.g. `` is sound only sound if a person hears it ? '' The most immediate philosophical topic that the riddle introduces involves the existence of the tree (
and the sound it produces ) outside of human perception . If no one is around to see , hear , touch or smell the tree , how could it be said to exist ? What is
it to say that it exists when such an existence is unknown ? Of course , from a scientific viewpoint , it exists . It is human beings that are able to perceive
it . George Berkeley in the 18th century developed subjective idealism , a metaphysical theory to respond to these questions , coined famously as `` to be is to
be perceived '' . Today meta - physicists are split . According to substance theory , a substance is distinct from its properties , while according to bundle
theory , an object is merely its sense data . The definition of sound , simplified , is a hearable noise . The tree will make a sound , even if nobody heard it
. The definition states that sound is a hearable noise . So the tree could have been heard , though nobody was around to do so .
</item 1>

<item 2>
If a tree falls in a forest
If there be no ears to hear , there will be no sound . '' Albert Einstein is reported to have asked his fellow physicist and friend Niels Bohr , one of the
founding fathers of quantum mechanics , whether he realistically believed that ' the moon does not exist if nobody is looking at it . ' To this Bohr replied
that however hard he ( Einstein ) may try , he would not be able to prove that it does , thus giving the entire riddle the status of a kind of an infallible
conjecture -- one that can not be either proved or disproved . The current phrasing appears to have originated in the 1910 book Physics by Charles Riborg Mann
and George Ransom Twiss . The question `` When a tree falls in a lonely forest , and no animal is near by to hear it , does it make a sound ? Why ? '' is posed
along with many other questions to quiz readers on the contents of the chapter , and as such , is posed from a purely physical point of view . Metaphysics (
edit ) The possibility of unperceived existence ( edit ) Main article : George Berkeley Can something exist without being perceived ? -- e.g.
</item 2>

<item 3>
If a tree falls in a forest
`` The mind is like akasa ( space ) . Just as there are the objects in the akasa , so there are thoughts in the mind . The akasa is the counterpart of the mind
and objects are of thought . One can not hope to measure the universe and study the phenomena . It is impossible . For the objects are mental creations ... The
universe is only an object created by the mind and has its being in the mind . '' ( Talks with Sri Ramana Maharshi , pp. 478 -- 479 ) . `` The shrutis ( a
subdivision of the Vedas ) and the sages say that the objects are only mental creations . They have no substantive being . Investigate the matter and ascertain
the truth of the statement . The result will be the conclusion that the objective world is in the subjective consciousness . '' ( TWSRM , p. 479 ) . In popular
culture ( edit ) Canadian songwriter , social activist and environmentalist Bruce Cockburn poses the question in the chorus of his 1989 song `` If a Tree Falls
'' and frames it with his lyrics as a pressing question regarding the cause and effect of deforestation .
</item 3>

<item 4>
If a tree falls in a forest
So then , how is it known that ' sound as it is mechanically understood ' will occur if that sound is not perceived ? Buddhist perspectives ( edit ) Tibetan (
edit ) from The Thirty - Seven Bodhisattva Practices by Ngulchu Thogme Zangpo , `` 22 Appearances are one 's own mind . From the beginning , mind 's nature is
free from the extremes of elaboration . Knowing this , not to engage the mind in subject - object duality is the bodhisattva 's practice . '' Zen ( edit ) Hui -
neng 's flag ( edit ) There is a well - known story of Hui - neng , a well - respected Buddhist monk who later became known as the founder of the Zen school ,
who one day happened to be passing by two monks . `` Two monks were arguing about the temple flag waving in the wind . One said , `` The flag moves . '' The
other said , `` The wind moves . '' They argued back and forth but could not agree . The Sixth Ancestor said , `` Gentlemen !
</item 4>

<item 5>
If a tree falls in a forest
It may then be recognized that the unobserved event was absolutely identical to an event which did not occur at all . '' ( this apparent quote has no
attribution or reference and none can be found online with reasonable effort ) . Of course , the fact that the tree is known to have changed state from '
upright ' to ' fallen ' implies that the event must be observed to ask the question at all -- even if only by the supposed deaf onlooker . The British
philosopher of science Roy Bhaskar , credited with developing critical realism has argued , in apparent reference to this riddle , that : If men ceased to exist
sound would continue to travel and heavy bodies to fall to the earth in exactly the same way , though ex hypothesi there would be no - one to know it This
existence of an unobserved real is integral to Bhaskar 's ontology , which contends ( in opposition to the various strains of positivism which have dominated
both natural and social science in the twentieth century ) that ' real structures exist independently of and are often out of phase with the actual patterns of
events ' .
</item 5>

<item 6>
If a tree falls in a forest
Washington - state - based wilderness conservatory Northwest Trek used a shortened form of the quote in its mid-1970s television advertisement , as such : ``
There is no sound unless someone is there to see it or hear it . Experience it at Northwest Trek . '' See also ( edit ) Counterfactual definiteness Descartes '
dream argument Epistemology Object ( philosophy ) Observer effect ( physics ) Kōan Ontology Schrödinger 's cat Principle of locality References ( edit ) Jump up
^ A Treatise Concerning the Principles of Human Knowledge , 1734 . section 23 . Jump up ^ A Treatise Concerning the Principles of Human Knowledge , 1734 .
section 45 . Jump up ^ Whether Berkeley thought objects continued to exist when unperceived by finite minds , and if so , in what manner , is the subject of
serious debate among Berkeley scholars . Jump up ^ The Chautauquan , June 1883 , Volume 3 , Issue 9 , p. 543 Jump up ^ Scientific American , April 5 , 1884 , pg
218 .
</item 6>

<item 7>
If a tree falls in a forest
It is unknown whether the source of this question is Berkeley or not . In June 1883 in the magazine The Chautauquan , the question was put , `` If a tree were
to fall on an island where there were no human beings would there be any sound ? '' They then went on to answer the query with , `` No . Sound is the sensation
excited in the ear when the air or other medium is set in motion . '' This seems to imply that the question is posed not from a philosophical viewpoint , but
from a purely scientific one . The magazine Scientific American corroborated the technical aspect of this question , while leaving out the philosophic side , a
year later when they asked the question slightly reworded , `` If a tree were to fall on an uninhabited island , would there be any sound ? '' And gave a more
technical answer , `` Sound is vibration , transmitted to our senses through the mechanism of the ear , and recognized as sound only at our nerve centers . The
falling of the tree or any other disturbance will produce vibration of the air . If there be no ears to hear , there will be no sound . ''
</item 7>

<item 8>
If a tree falls in a forest
2.1 Hui - neng 's Flag 2.4. 2.2 Hui - neng Receives The Dharma 2.5 Hinduism 3 In popular culture 4 See also 5 References History ( edit ) Philosopher George
Berkeley , in his work , A Treatise Concerning the Principles of Human Knowledge ( 1710 ) , proposes , `` But , say you , surely there is nothing easier than
for me to imagine trees , for instance , in a park ( ... ) and nobody by to perceive them . ( ... ) The objects of sense exist only when they are perceived ;
the trees therefore are in the garden ( ... ) no longer than while there is somebody by to perceive them . '' ( It is worth noting that the quote from section
45 is arguably a statement of an objection to Berkeley 's view , and not a proclamation of it . ) Nevertheless , Berkeley never actually wrote about the
question . Some years later , a similar question is posed . It is unknown whether the source of this question is Berkeley or not .
</item 8>

<item 9>
If a tree falls in a forest
155 -- 158 . Jump up ^ Watts 1962 , pp. 111 -- 113 Jump up ^ A Buddhist Bible , `` Sutra Spoken by The Sixth Patriarch '' Retrieved from ``
https://en.wikipedia.org/w/index.php?title=If_a_tree_falls_in_a_forest&oldid=812540716 '' Categories : Fictional trees Riddles Philosophical problems Thought
experiments in philosophy of mind Talk Contents About Wikipedia Español فارسی עברית Македонски Nederlands Português Русский Svenska ไทย 中文 Edit links This page
was last edited on 28 November 2017 , at 12 : 14 . Text is available under the Creative Commons Attribution - ShareAlike License ; additional terms may apply .
By using this site , you agree to the Terms of Use and Privacy Policy . Wikipedia ® is a registered trademark of the Wikimedia Foundation , Inc. , a non-profit
organization . About Wikipedia
</item 9>

<item 10>
If a tree falls in a forest
If a tree falls in a forest - wikipedia If a tree falls in a forest Jump to : navigation , search For the 2011 documentary , see If a Tree Falls : A Story of
the Earth Liberation Front . A fallen tree in a forest `` If a tree falls in a forest and no one is around to hear it , does it make a sound ? '' is a
philosophical thought experiment that raises questions regarding observation and perception . Contents ( hide ) 1 History 2 Metaphysics 2.1 The possibility of
unperceived existence 2.2 Knowledge of the unobserved world 2.3 The dissimilarity between sensation and reality 2.4 Buddhist perspectives 2.4. 1 Tibetan 2.4. 2
Zen 2.4. 2.1 Hui - neng 's Flag 2.4.
</item 10>
=== END ITEM id=349099f4b2626778 ===

=== ITEM id=5ec3a442feafd390 ===
Gold sentences and facts:
Sentence 0: Operating System Version ( s ) Location Unix , Unix - like , POSIX / etc / hosts Microsoft Windows 3.1 % WinDir % \ HOSTS 95 , 98 , ME % WinDir % \ hosts NT , 2000 , XP , 2003 , Vista , 2008 , 7 , 2012 , 8 , 10 % SystemRoot % \ System32 \ drivers \ etc \ hosts Windows Mobile , Windows Phone Registry key under HKEY_LOCAL_MACHINE \ Comm \ Tcpip \ Hosts Apple Macintosh 9 and earlier Preferences or System folder Mac OS X 10.0 -- 10.1. 5 ( Added through NetInfo or niload ) Mac OS X 10.2 and newer / etc / hosts ( a symbolic link to / private / etc / hosts ) Novell NetWare SYS : etc \ hosts OS / 2 & eComStation `` bootdrive '' : \ mptn \ etc \ Symbian Symbian OS 6.1 -- 9.0 C : \ system \ data \ hosts Symbian OS 9.1 + C : \ private \ 10000882 \ hosts MorphOS NetStack ENVARC : sys / net / hosts AmigaOS < 4 AmiTCP : db / hosts DEVS : Internet / hosts AROS ENVARC : AROSTCP / db / hosts Android / etc / hosts ( a symbolic link to / system / etc / hosts ) iOS iOS 2.0 and newer / etc / hosts ( a symbolic link to / private / etc / hosts ) TOPS - 20 < SYSTEM > HOSTS. TXT Plan 9 / lib / ndb / hosts BeOS / boot / beos / etc / hosts Haiku / boot / common / settings / network / hosts OpenVMS UCX UCX $ HOST TCPware TCPIP $ HOST RISC OS 3.7 , 5 ! Boot. Resources. ! Internet. files. Hosts later boot sequence ! Boot. Choices. Hardware. Disabled. Internet. Files. Hosts
  Fact 0.0: The hosts file on Unix, Unix-like, and POSIX systems is at /etc/hosts
  Fact 0.1: On Microsoft Windows 3.1 the hosts file is at %WinDir%\HOSTS
  Fact 0.2: On Windows 95, 98, and ME the hosts file is at %WinDir%\hosts
  Fact 0.3: On Windows NT, 2000, XP, 2003, Vista, 2008, 7, 2012, 8, and 10 the hosts file is at %SystemRoot%\System32\drivers\etc\hosts
  Fact 0.4: On Windows Mobile and Windows Phone the hosts entry is a registry key under HKEY_LOCAL_MACHINE\Comm\Tcpip\Hosts
  Fact 0.5: On Apple Macintosh 9 and earlier the hosts file is in the Preferences or System folder
  Fact 0.6: On Mac OS X 10.0 to 10.1.5 the hosts file was added through NetInfo or niload
  Fact 0.7: On Mac OS X 10.2 and newer the hosts file is /etc/hosts, a symbolic link to /private/etc/hosts
  Fact 0.8: On Novell NetWare the hosts file is at SYS:etc\hosts
  Fact 0.9: On OS/2 and eComStation the hosts file is at the bootdrive path \mptn\etc\
  Fact 0.10: On Symbian OS 6.1 to 9.0 the hosts file is at C:\system\data\hosts
  Fact 0.11: On Symbian OS 9.1 and later the hosts file is at C:\private\10000882\hosts
  Fact 0.12: On MorphOS (NetStack) the hosts file is at ENVARC:sys/net/hosts
  Fact 0.13: On AmigaOS before 4 the hosts file is at AmiTCP:db/hosts or DEVS:Internet/hosts
  Fact 0.14: On AROS the hosts file is at ENVARC:AROSTCP/db/hosts
  Fact 0.15: On Android the hosts file is /etc/hosts, a symbolic link to /system/etc/hosts
  Fact 0.16: On iOS 2.0 and newer the hosts file is /etc/hosts, a symbolic link to /private/etc/hosts
  Fact 0.17: On TOPS-20 the hosts file is <SYSTEM>HOSTS.TXT
  Fact 0.18: On Plan 9 the hosts file is /lib/ndb/hosts
  Fact 0.19: On BeOS the hosts file is /boot/beos/etc/hosts
  Fact 0.20: On Haiku the hosts file is /boot/common/settings/network/hosts
  Fact 0.21: On OpenVMS with UCX the hosts file is UCX$HOST
  Fact 0.22: On OpenVMS with TCPware the hosts file is TCPIP$HOST
  Fact 0.23: On RISC OS 3.7 and 5 the hosts file is at !Boot.Resources.!Internet.files.Hosts, loaded in the later boot sequence
  Fact 0.24: On RISC OS the disabled hosts file is at !Boot.Choices.Hardware.Disabled.Internet.Files.Hosts

Retrieved items:
<item 1>
hosts (file)
History of BIND . Retrieved 2017 - 07 - 01 . Jump up ^ `` Cisco Networking Academy Program : First - Year Companion Guide '' , Cisco Systems , Inc. , 2002 ( 2nd
Edition ) , page 676 , ISBN 1 - 58713 - 025 - 4 Jump up ^ `` Linux Network Administrators Guide : Writing hosts and networks files '' . Retrieved May 16 , 2010
. Jump up ^ `` Hosts File '' . Retrieved August 10 , 2011 . Jump up ^ `` Microsoft KB Q314053 : TCP / IP and NBT configuration parameters for Windows XP '' .
Retrieved August 28 , 2010 . Jump up ^ `` Microsoft KB 972034 Revision 2.0 : default hosts files '' . Retrieved August 28 , 2010 . ^ Jump up to : `` Mac OS X :
How to Add Hosts to Local Hosts File '' . Retrieved August 28 , 2010 . ^ Jump up to : `` The Haiku / BeOS Tip Server '' . Retrieved November 30 , 2012 .
</item 1>

<item 2>
hosts (file)
It is usually named hosts , without an extension . Operating System Version ( s ) Location Unix , Unix - like , POSIX / etc / hosts Microsoft Windows 3.1 %
WinDir % \ HOSTS 95 , 98 , ME % WinDir % \ hosts NT , 2000 , XP , 2003 , Vista , 2008 , 7 , 2012 , 8 , 10 % SystemRoot % \ System32 \ drivers \ etc \ hosts
Windows Mobile , Windows Phone Registry key under HKEY_LOCAL_MACHINE \ Comm \ Tcpip \ Hosts Apple Macintosh 9 and earlier Preferences or System folder Mac OS X
10.0 -- 10.1.
</item 2>

<item 3>
hosts (file)
Retrieved November 30 , 2012 . Jump up ^ RISC OS 6.14 Jump up ^ `` Remove Trojan. Qhosts -- Symantec '' . Retrieved May 16 , 2010 . External links ( edit )
Windows TCP / IP in Windows 2000 Professional Microsoft TCP / IP Host Name Resolution Order Blocking Unwanted Connections with a Hosts File ( Creative Commons
License ) Solution if Windows stops you from modifying the Hosts file or the Lmhosts file in Windows Vista / 7 / 8 Beginners Guide on Editing Hosts File List of
Microsoft related domains that Windows wo n't redirect Mac How to block unwanted domains with a HOSTS file on Mac OS X 10.2 or later `` Gas Mask '' - Hosts file
manager for Mac OS X Linux Block unwanted advertisements with / etc / hosts file on Linux Retrieved from `` https://en.wikipedia.org/w/index.
</item 3>

<item 4>
hosts (file)
5 ( Added through NetInfo or niload ) Mac OS X 10.2 and newer / etc / hosts ( a symbolic link to / private / etc / hosts ) Novell NetWare SYS : etc \ hosts OS /
2 & eComStation `` bootdrive '' : \ mptn \ etc \ Symbian Symbian OS 6.1 -- 9.0 C : \ system \ data \ hosts Symbian OS 9.1 + C : \ private \ 10000882 \ hosts
MorphOS NetStack ENVARC : sys / net / hosts AmigaOS < 4 AmiTCP : db / hosts DEVS : Internet / hosts AROS ENVARC : AROSTCP / db / hosts Android / etc / hosts ( a
symbolic link to / system / etc / hosts ) iOS iOS 2.0 and newer / etc / hosts ( a symbolic link to / private / etc / hosts ) TOPS - 20 < SYSTEM > HOSTS.
</item 4>

<item 5>
hosts (file)
TXT Plan 9 / lib / ndb / hosts BeOS / boot / beos / etc / hosts Haiku / boot / common / settings / network / hosts OpenVMS UCX UCX $ HOST TCPware TCPIP $ HOST
RISC OS 3.7 , 5 ! Boot. Resources. ! Internet. files. Hosts later boot sequence ! Boot. Choices. Hardware. Disabled. Internet. Files. Hosts History ( edit ) The
ARPANET , the predecessor of the Internet , had no distributed host name database . Each network node maintained its own map of the network nodes as needed and
assigned them names that were memorable to the users of the system . There was no method for ensuring that all references to a given node in a network were
using the same name , nor was there a way to read the hosts file of another computer to automatically obtain a copy . The small size of the ARPANET kept the
administrative overhead small to maintain an accurate hosts file . Network nodes typically had one address and could have many names .
</item 5>

<item 6>
hosts (file)
Comment lines may be included ; they are indicated by an octothorpe ( # ) in the first position of such lines . Entirely blank lines in the file are ignored .
For example , a typical hosts file may contain the following : 127.0. 0.1 localhost loopback : : 1 localhost This example only contains entries for the loopback
addresses of the system and their host names , a typical default content of the hosts file . The example illustrates that an IP address may have multiple host
names ( localhost and loopback ) , and that a host name may be mapped to both IPv4 and IPv6 IP addresses , as shown on the first and second lines respectively .
Location in the file system ( edit ) The location of the hosts file in the file system hierarchy varies by operating system . It is usually named hosts ,
without an extension .
</item 6>

<item 7>
hosts (file)
It is a common part of an operating system 's Internet Protocol ( IP ) implementation , and serves the function of translating human - friendly hostnames into
numeric protocol addresses , called IP addresses , that identify and locate a host in an IP network . In some operating systems , the contents of the hosts file
is used preferentially to other name resolution methods , such as the Domain Name System ( DNS ) , but many systems implement name service switches , e.g. ,
nsswitch. conf for Linux and Unix , to provide customization . Unlike remote DNS resolvers , the hosts file is under the direct control of the local computer 's
administrator . File content ( edit ) This section needs additional citations for verification . Please help improve this article by adding citations to
reliable sources . Unsourced material may be challenged and removed . ( April 2017 ) ( Learn how and when to remove this template message ) The hosts file
contains lines of text consisting of an IP address in the first text field followed by one or more host names . Each field is separated by white space -- tabs
are often preferred for historical reasons , but spaces are also used .
</item 7>

<item 8>
hosts (file)
Network nodes typically had one address and could have many names . As local area TCP / IP computer networks gained popularity , however , the maintenance of
hosts files became a larger burden on system administrators as networks and network nodes were being added to the system with increasing frequency .
Standardization efforts , such as the format specification of the file HOSTS. TXT in RFC 952 , and distribution protocols , e.g. , the hostname server described
in RFC 953 , helped with these problems , but the centralized and monolithic nature of hosts files eventually necessitated the creation of the distributed
Domain Name System ( DNS ) . On some old systems a file named networks is present that has similar to hosts file functions containing names of networks .
Extended applications ( edit ) In its function of resolving host names , the hosts file may be used to define any hostname or domain name for use in the local
system . Redirecting local domains Some web service and intranet developers and administrators define locally defined domains in a LAN for various purposes ,
such as accessing the company 's internal resources or to test local websites in development .
</item 8>

<item 9>
hosts (file)
Internet resource blocking Entries in the hosts file may be used to block online advertising , or the domains of known malicious resources and servers that
contain spyware , adware , and other malware . This may be achieved by adding entries for those sites to redirect requests to another address that does not
exist or to a harmless destination . Commercial software applications may be used to populate the hosts file with entries of known undesirable Internet
resources automatically . Security issues ( edit ) The hosts file may present an attack vector for malicious software . The file may be modified , for example ,
by adware , computer viruses , or trojan horse software to redirect traffic from the intended destination to sites hosting malicious or unwanted content . The
widespread computer worm Mydoom. B blocked users from visiting sites about computer security and antivirus software and also affected access from the
compromised computer to the Microsoft Windows Update website . See also ( edit ) DNSBL , a DNS - based blackhole list Content - control software Ad filtering
References ( edit ) Jump up ^ Internet Systems Consortium . `` The Most Widely Used Name Server Software : BIND '' . History of BIND . Retrieved 2017 - 07 - 01
.
</item 9>

<item 10>
hosts (file)
Hosts ( file ) - wikipedia Hosts ( file ) Jump to : navigation , search For other uses , see Host ( disambiguation ) . The computer file hosts is an operating
system file that maps hostnames to IP addresses . It is a plain text file . Originally a file named HOSTS. TXT was manually maintained and made available via
file sharing by Stanford Research Institute for the ARPANET membership , containing the hostnames and address of hosts as contributed for inclusion by member
organizations . The Domain Name System , first described in 1983 and implemented in 1984 , automated the publication process and provided instantaneous and
dynamic hostname resolution in the rapidly growing network . In modern operating systems , the hosts file remains an alternative name resolution mechanism ,
configurable often as part of facilities such as the Name Service Switch as either the primary method or as a fallback method . Contents ( hide ) 1 Purpose 2
File content 3 Location in the file system 4 History 5 Extended applications 6 Security issues 7 See also 8 References 9 External links Purpose ( edit ) The
hosts file is one of several system facilities that assists in addressing network nodes in a computer network .
</item 10>
=== END ITEM id=5ec3a442feafd390 ===

=== ITEM id=6c197ce942cccef3 ===
Gold sentences and facts:
Sentence 0: The Roosevelt Corollary was an addition to the Monroe Doctrine articulated by President Theodore Roosevelt in his State of the Union address in 1904 after the Venezuela Crisis of 1902 -- 03 . The corollary states that the United States will intervene in conflicts between European countries and Latin American countries to enforce legitimate claims of the European powers , rather than having the Europeans press their claims directly .
  Fact 0.0: The Roosevelt Corollary was an addition to the Monroe Doctrine.
  Fact 0.1: President Theodore Roosevelt articulated the Roosevelt Corollary.
  Fact 0.2: Theodore Roosevelt articulated the Roosevelt Corollary in his State of the Union address in 1904.
  Fact 0.3: The Roosevelt Corollary was articulated after the Venezuela Crisis of 1902-03.
  Fact 0.4: The Roosevelt Corollary states that the United States will intervene in conflicts between European countries and Latin American countries.
  Fact 0.5: Under the Roosevelt Corollary, the United States intervenes to enforce legitimate claims of the European powers.
  Fact 0.6: Under the Roosevelt Corollary, the United States intervenes rather than having the Europeans press their claims directly.

Retrieved items:
<item 1>
List of cities in the Americas by year of foundation
Founded in 1498 , by Bartholomew Columbus . 1502 Santa Cruz del Seibo El Seibo Dominican Republic 1504 Azua de Compostela Azua Dominican Republic 1504 Santiago
de los Caballeros Santiago Dominican Republic 1505 Cotuí Sánchez Ramírez Dominican Republic 1506 Salvaleón de Higüey La Altagracia Dominican Republic 1508 Bonao
Monseñor Nouel Dominican Republic 1508 Caparra Puerto Rico United States First city founded by Europeans , although not continuously inhabited , in the United
States . Abandoned in 1521 with the removal of the capital to San Juan . 1510 Santa Maria la Antigua del Darien Urabá Colombia First city founded by Europeans
on the continent of South America . 1510 Nombre de Dios Colon Panama Oldest continuously inhabited European established settlement in Panama . 1511 Baracoa
Guantánamo Cuba Oldest continuously inhabited European established settlement in Cuba , former capital of Cuba .
</item 1>

<item 2>
Roosevelt Corollary
Econ Journal Watch , 4 ( 1 ) , 3 - 45 . Glickman , Robert Jay . Norteamérica vis - à - vis Hispanoamérica : ¿ opposición o asociación ? Toronto : Canadian
Academy of the Arts , 2005 . ISBN 0 - 921907 - 09 - 5 . Meiertöns , Heiko ( 2010 ) The Doctrines of US Security Policy - An Evaluation under International Law ,
Cambridge University Press , ISBN 978 - 0 - 521 - 76648 - 7 . Mitchell , Nancy . The Danger of Dreams : German and American Imperialism in Latin America ( 1999
) , Mitchener , Kris James , and Marc Weidenmier . `` Empire , public goods , and the Roosevelt Corollary . '' Journal of Economic History ( 2005 ) 64 # 5 pp
658 + Rabe , Stephen G. `` Theodore Roosevelt , the Panama Canal and the Roosevelt Corollary : Sphere of Influence Diplomacy , '' ch 16 in Serge Ricard , ed.
</item 2>

<item 3>
Roosevelt Corollary
The Roosevelt Corollary was an addition to the Monroe Doctrine articulated by President Theodore Roosevelt in his State of the Union address in 1904 after the
Venezuela Crisis of 1902 -- 03 . The corollary states that the United States will intervene in conflicts between European countries and Latin American countries
to enforce legitimate claims of the European powers , rather than having the Europeans press their claims directly . Roosevelt tied his policy to the Monroe
Doctrine , and it was also consistent with his foreign policy included in his Big Stick Diplomacy . Roosevelt stated that in keeping with the Monroe Doctrine ,
the United States was justified in exercising `` international police power '' to put an end to chronic unrest or wrongdoing in the Western Hemisphere . While
the Monroe Doctrine had sought to prevent European intervention , the Roosevelt Corollary was used to justify US intervention throughout the hemisphere . In
1934 , President Franklin D. Roosevelt renounced interventionism and established his Good Neighbor policy for the Western Hemisphere .
</item 3>

<item 4>
Roosevelt Corollary
In 1934 , Franklin D. Roosevelt further renounced interventionism and established his `` Good Neighbor policy '' that led to the annulment of the Platt
Amendment by the Treaty of Relations with Cuba in 1934 , and the negotiation of compensation for Mexico 's nationalization of foreign - owned oil assets in 1938
. Indeed , leaving unchallenged the emergence of dictatorships like that of Fulgencio Batista in Cuba , Rafael Leonidas Trujillo in the Dominican Republic ,
Anastasio Somoza in Nicaragua , and François Duvalier in Haiti were each considered to be `` Frankenstein dictators '' due to the mishandlings of the American
occupations in the countries . The era of the Good Neighbor Policy ended with the ramp - up of the Cold War in 1945 , as the United States felt there was a
greater need to protect the western hemisphere from Soviet influence . In 1954 , Secretary of State John Foster Dulles invoked the Monroe Doctrine and the
Roosevelt Corollary at the Tenth Pan-American Conference in Caracas , denouncing the intervention of Soviet Communism in Guatemala .
</item 4>

<item 5>
Roosevelt Corollary
Contents ( hide ) 1 Use 1.1 Shift to the `` Good Neighbor '' policy 2 Criticism 3 See also 4 References 5 Bibliography Use ( edit ) See also : Overseas
interventions of the United States The Roosevelt Corollary was an addition to the Monroe Doctrine ; however , it could be seen as a departure . While the Monroe
Doctrine said European countries should stay out of Latin America , the Roosevelt Corollary took this further to say he had the right to exercise military force
in Latin American countries to keep European countries out . Historian Walter LaFeber wrote ( Roosevelt ) essentially turns the Monroe Doctrine on its head and
toes says the Europeans should stay out , but the United States has the right , under the doctrine , to go in to exercise police power to keep the Europeans out
of the way .
</item 5>

<item 6>
Roosevelt Corollary
Latin America -- United States relations Retrieved from `` https://en.wikipedia.org/w/index.php?title=Roosevelt_Corollary&oldid=815269793 '' Categories :
Foreign policy doctrines of the United States Hegemony Banana Wars Presidency of Theodore Roosevelt History of the United States ( 1865 -- 1918 ) History of
United States expansionism 1904 in international relations 1904 in the United States Hidden categories : CS1 errors : dates Use dmy dates from March 2011 Talk
Contents About Wikipedia Deutsch Español Esperanto Euskara Français 贛 語 한국어 Bahasa Indonesia Italiano Polski Português Edit links This page was last edited on
13 December 2017 , at 20 : 57 . Text is available under the Creative Commons Attribution - ShareAlike License ; additional terms may apply . By using this site
, you agree to the Terms of Use and Privacy Policy . Wikipedia ® is a registered trademark of the Wikimedia Foundation , Inc. , a non-profit organization .
About Wikipedia
</item 6>

<item 7>
Cuban War of Independence
Joseph E. Wisan wrote in an essay titled `` The Cuban Crisis As Reflected In The New York Press '' , published in `` American Imperialism '' in 1898 : `` In the
opinion of the writer , the Spanish -- American War would not have occurred had not the appearance of Hearst in New York journalism precipitated a bitter battle
for newspaper circulation . '' It has also been argued that the main reason the United States entered the war was its failed attempt to purchase Cuba from Spain
. Further information : Propaganda of the Spanish -- American War Hostilities started hours after the declaration of war when a contingent of US Navy ships
under Admiral William T. Sampson blockaded several Cuban ports . The Americans decided to invade Cuba and to start in Oriente , where the Cubans had almost
absolute control . They cooperated by establishing a beachhead and protecting the U.S. landing in Daiquiri . The first U.S. objective was to capture the city of
Santiago de Cuba in order to destroy Linares ' army and Cervera 's fleet .
</item 7>

<item 8>
American entry into World War I
This sentiment played a major role in arousing fear of Germany , and suspicions regarding everyone of German descent who could not `` prove '' 100 % loyalty .
Preparedness movement ( edit ) This section needs additional citations for verification . Please help improve this article by adding citations to reliable
sources . Unsourced material may be challenged and removed . ( September 2017 ) ( Learn how and when to remove this template message ) Main article :
Preparedness Movement By 1915 , Americans were paying much more attention to the war . The sinking of the Lusitania had a strong effect on public opinion
because of the deaths of American civilians . That year , a strong `` Preparedness '' movement emerged . Proponents argued that the United States needed to
immediately build up strong naval and land forces for defensive purposes ; an unspoken assumption was that America would fight sooner or later .
</item 8>

<item 9>
Roosevelt Corollary
Serge Ricard of the University of Paris goes even further , stating that the Roosevelt Corollary was not merely an addendum to the earlier Monroe Doctrine ,
through which the U.S. pledged to protect the Americas from European imperialist interventions . Rather , the Roosevelt Corollary was `` an entirely new
diplomatic tenet that epitomized his ' big stick ' approach to foreign policy . '' In other words , while the Monroe Doctrine sought to bar entry to the
European empires , the Roosevelt Corollary announced the United States intention to take their place . See also ( edit ) International relations portal
Government of the United States portal United States portal Wikisource has original text related to this article : Roosevelt Corollary History of the United
States ( 1865 -- 1918 ) United States territorial acquisitions New Imperialism Dollar Diplomacy Big Stick ideology Drago Doctrine Bush Doctrine References (
edit ) Jump up ^ Theodore Roosevelt , 26th President . Prod . David Grubin . By David Grubin and Geoffrey C. Ward . Perf . Walter LaFeber . David Grubin
Productions , Inc. , 1996 .
</item 9>

<item 10>
American entry into World War I
To make the conflict seem like a better idea , he painted the conflict idealistically , stating that the war would `` make the world safe for democracy '' and
later that it would be a `` war to end war '' . The United States had a moral responsibility to enter the war , Wilson proclaimed . The future of the world was
being determined on the battlefield , and American national interest demanded a voice . Wilson 's definition of the situation won wide acclaim , and , indeed ,
has shaped America 's role in world and military affairs ever since . Wilson believed that if the Central Powers won , the consequences would be bad for the
United States . Germany would have dominated the continent and perhaps would gain control of the seas as well . Latin America could well have fallen under
Berlin 's control . The dream of spreading democracy , liberalism , and independence would have been shattered . On the other hand , if the Allies had won
without help , there was a danger they would carve up the world without regard to American commercial interests . They were already planning to use government
subsidies , tariff walls , and controlled markets to counter the competition posed by American businessmen .
</item 10>
=== END ITEM id=6c197ce942cccef3 ===

=== ITEM id=1eea8d3be9705be7 ===
Gold sentences and facts:
Sentence 0: Because the states were preexisting political entities , the U.S. Constitution did not need to define or explain federalism in any one section but it often mentions the rights and responsibilities of state governments and state officials in relation to the federal government . The federal government has certain express powers ( also called enumerated powers ) which are powers spelled out in the Constitution , including the right to levy taxes , declare war , and regulate interstate and foreign commerce . In addition , the Necessary and Proper Clause gives the federal government the implied power to pass any law `` necessary and proper '' for the execution of its express powers . Other powers -- the reserved powers -- are reserved to the people or the states . The power delegated to the federal government was significantly expanded by the Supreme Court decision in McCulloch v. Maryland ( 1819 ) , amendments to the Constitution following the Civil War , and by some later amendments -- as well as the overall claim of the Civil War , that the states were legally subject to the final dictates of the federal government .
  Fact 0.0: The states were preexisting political entities.
  Fact 0.1: The U.S. Constitution did not need to define or explain federalism in any one section, because the states were preexisting political entities.
  Fact 0.2: The U.S. Constitution often mentions the rights and responsibilities of state governments and state officials in relation to the federal government.
  Fact 0.3: The federal government has certain express powers, also called enumerated powers.
  Fact 0.4: Express powers are powers spelled out in the Constitution.
  Fact 0.5: The federal government's express powers include the right to levy taxes.
  Fact 0.6: The federal government's express powers include the right to declare war.
  Fact 0.7: The federal government's express powers include the right to regulate interstate commerce.
  Fact 0.8: The federal government's express powers include the right to regulate foreign commerce.
  Fact 0.9: The Necessary and Proper Clause gives the federal government the implied power to pass any law 'necessary and proper' for the execution of its express powers.
  Fact 0.10: Other powers, the reserved powers, are reserved to the people or the states.
  Fact 0.11: The Supreme Court decision in McCulloch v. Maryland (1819) significantly expanded the power delegated to the federal government.
  Fact 0.12: Amendments to the Constitution following the Civil War significantly expanded the power delegated to the federal government.
  Fact 0.13: Some later amendments to the Constitution significantly expanded the power delegated to the federal government.
  Fact 0.14: The overall claim of the Civil War, that the states were legally subject to the final dictates of the federal government, significantly expanded the power delegated to the federal government.

Retrieved items:
<item 1>
Federalism
In 1995 the Supreme Court rejected the Gun - Free School Zones Act in the Lopez decision , and also rejected the civil remedy portion of the Violence Against
Women Act of 1994 in the United States v. Morrison decision . Recently , the Commerce Clause was interpreted to include marijuana laws in the Gonzales v. Raich
decision . Dual federalism holds that the federal government and the state governments are co-equals , each sovereign . However , since the Civil War Era , the
national courts often interpret the federal government as the final judge of its own powers under dual federalism . The establishment of Native American
governments ( which are separate and distinct from state and federal government ) exercising limited powers of sovereignty , has given rise to the concept of ``
bi-federalism . '' Venezuela ( edit ) The Federal War ended in 1863 with the signing of the Treaty of Coche by both the centralist government of the time and
the Federal Forces . The United States of Venezuela were subsequently incorporated under a `` Federation of Sovereign States '' upon principles borrowed from
the Articles of Confederation of the United States of America .
</item 1>

<item 2>
Federalism
Its distinctive feature , exemplified in the founding example of modern federalism by the United States of America under the Constitution of 1787 , is a
relationship of parity between the two levels of government established . It can thus be defined as a form of government in which there is a division of powers
between two levels of government of equal status . Federalism differs from confederalism , in which the general level of government is subordinate to the
regional level , and from devolution within a unitary state , in which the regional level of government is subordinate to the general level . It represents the
central form in the pathway of regional integration or separation , bounded on the less integrated side by confederalism and on the more integrated side by
devolution within a unitary state . Leading examples of the federation or federal state include the United States , Canada , Brazil , Germany , Switzerland ,
Argentina , Australia and India . Some also today characterize the European Union as the pioneering example of federalism in a multi-state setting , in a
concept termed the federal union of states . Contents ( hide ) 1 Overview 1.1 Explanations for adoption of federalist systems 2 European vs.
</item 2>

<item 3>
Federalism
See also ( edit ) Consociationalism Cooperative federalism Democratic World Federalists Federal Union Layer cake federalism Pillarisation States ' rights Union
of Utrecht World Federalist Movement Notes and references ( edit ) Jump up ^ In 1946 , scholar Kenneth Wheare observed that the two levels of government in the
US were ' co-equally supreme ' . In this he echoed the perspective of the founding fathers , James Madison in Federalist 39 having seen the several states as
forming ' distinct and independent portions of the supremacy ' in relation to the general government . Wheare , Kenneth ( 1946 ) Federal Government , Oxford
University Press , London , pp. 10 - 15 . Madison , James , Hamilton , Alexander and Jay , John ( 1987 ) The Federalist Papers , Penguin , Harmondsworth , p.
258 . Jump up ^ Law , John ( 2013 ) ' How Can We Define Federalism ? ' , in Perspectives on Federalism , Vol. 5 , No. 3 , pp. E105 - 6 .
</item 3>

<item 4>
Federalism
The federalists , on the other hand , argued that it was impossible to list all the rights , and those that were not listed could be easily overlooked because
they were not in the official bill of rights . Rather , rights in specific cases were to be decided by the judicial system of courts . After the American Civil
War , the federal government increased greatly in influence on everyday life and in size relative to the state governments . Reasons included the need to
regulate businesses and industries that span state borders , attempts to secure civil rights , and the provision of social services . The federal government
acquired no substantial new powers until the acceptance by the Supreme Court of the Sherman Anti-Trust Act . From 1938 until 1995 , the U.S. Supreme Court did
not invalidate any federal statute as exceeding Congress ' power under the Commerce Clause . Most actions by the federal government can find some legal support
among the express powers , such as the Commerce Clause , whose applicability has been narrowed by the Supreme Court in recent years .
</item 4>

<item 5>
Federalism
The U.S. Constitution was written as a reaction to the Articles of Confederation , under which the United States was a loose confederation with a weak central
government . In contrast , Europe has a greater history of unitary states than North America , thus European `` federalism '' argues for a weaker central
government , relative to a unitary state . The modern American usage of the word is much closer to the European sense . As the power of the Federal government
has increased , some people have perceived a much more unitary state than they believe the Founding Fathers intended . Most people politically advocating ``
federalism '' in the United States argue in favor of limiting the powers of the federal government , especially the judiciary ( see Federalist Society , New
Federalism ) . In Canada , federalism typically implies opposition to sovereigntist movements ( most commonly Quebec separatism ) . The governments of Argentina
, Australia , Brazil , India , and Mexico , among others , are also organized along federalist principles . Federalism may encompass as few as two or three
internal divisions , as is the case in Belgium or Bosnia and Herzegovina .
</item 5>

<item 6>
Federalism
American government has evolved from a system of dual federalism to one of associative federalism . In `` Federalist No. 46 , '' James Madison asserted that the
states and national government `` are in fact but different agents and trustees of the people , constituted with different powers . '' Alexander Hamilton ,
writing in `` Federalist No. 28 , '' suggested that both levels of government would exercise authority to the citizens ' benefit : `` If their ( the peoples ' )
rights are invaded by either , they can make use of the other as the instrument of redress . '' ( 1 ) The United States is composed of fifty self - governing
states and several territories . Because the states were preexisting political entities , the U.S. Constitution did not need to define or explain federalism in
any one section but it often mentions the rights and responsibilities of state governments and state officials in relation to the federal government . The
federal government has certain express powers ( also called enumerated powers ) which are powers spelled out in the Constitution , including the right to levy
taxes , declare war , and regulate interstate and foreign commerce .
</item 6>

<item 7>
Federalism
- J. Proudhon , The Principle of Federation , 1863 . A Comparative Bibliography : Regulatory Competition on Corporate Law A Rhetoric for Ratification : The
Argument of the Federalist and its Impact on Constitutional Interpretation Brainstorming National ( in French ) Teaching about Federalism in the United States -
From the Education Resources Information Center Clearinghouse for Social Studies / Social Science Education Bloomington , Indiana . An Ottawa , Ontario , Canada
- based international organization for federal countries that share best practices among countries with that system of government Tenth Amendment Center
Federalism and States Rights in the U.S. BackStory Radio episode on the origins and current status of Federalism Constitutional law scholar Hester Lessard
discusses Vancouver 's Downtown Eastside and jurisdictional justice McGill University ,
</item 7>

<item 8>
Federalism
Jump up ^ See Law , John ( 2013 ) , p. 104 . http://www.on-federalism.eu/attachments/169_download.pdf This author identifies two distinct federal forms , where
before only one was known , based upon whether sovereignty ( conceived in its core meaning of ultimate authority ) resides in the whole ( in one people ) or in
the parts ( in many peoples ) . This is determined by the absence or presence of a right of secession for the parts . The structures are termed , respectively ,
the federal state ( or federation ) and the federal union of states ( or federal union ) . Jump up ^ Madison , James , Hamilton , Alexander and Jay , John (
1987 ) The Federalist Papers , Penguin , Harmondsworth , p. 259 . Jump up ^ Law , John ( 2012 ) ' Sense on Federalism ' , in Political Quarterly , Vol. 83 , No.
3 , p. 544 .
</item 8>

<item 9>
Federalism
In addition , the Necessary and Proper Clause gives the federal government the implied power to pass any law `` necessary and proper '' for the execution of its
express powers . Other powers -- the reserved powers -- are reserved to the people or the states . The power delegated to the federal government was
significantly expanded by the Supreme Court decision in McCulloch v. Maryland ( 1819 ) , amendments to the Constitution following the Civil War , and by some
later amendments -- as well as the overall claim of the Civil War , that the states were legally subject to the final dictates of the federal government . The
Federalist Party of the United States was opposed by the Democratic - Republicans , including powerful figures such as Thomas Jefferson . The Democratic -
Republicans mainly believed that : the Legislature had too much power ( mainly because of the Necessary and Proper Clause ) and that they were unchecked ; the
Executive had too much power , and that there was no check on the executive ; a dictator would arise ; and that a bill of rights should be coupled with the
constitution to prevent a dictator ( then believed to eventually be the president ) from exploiting or tyrannizing citizens .
</item 9>

<item 10>
Federalism
This was reaffirmed by the Tenth Amendment to the United States Constitution , which reserves all powers and rights that are not delegated to the Federal
Government as left to the States and to the people . Bicameralism ( edit ) The structures of most federal governments incorporate mechanisms to protect the
rights of component states . One method , known as ' intrastate federalism ' , is to directly represent the governments of component states in federal political
institutions . Where a federation has a bicameral legislature the upper house is often used to represent the component states while the lower house represents
the people of the nation as a whole . A federal upper house may be based on a special scheme of apportionment , as is the case in the senates of the United
States and Australia , where each state is represented by an equal number of senators irrespective of the size of its population .
</item 10>
=== END ITEM id=1eea8d3be9705be7 ===
