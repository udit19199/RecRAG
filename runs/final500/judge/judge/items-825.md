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

=== ITEM id=39bcc1a56b302c98 ===
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
Of the remaining 50 % , 23 % have taken steps to reform their 6 - day workweek through such measures as reducing the working week from 6 days to 5.5 days .
United states ( edit ) The standard working week in the United States begins on Monday and ends on Friday , 40 hours per week , with Saturday and Sunday being
weekend days . However , in practice , only 42 % of employees work 40 - hour weeks . The average workweek for full - time employees is 47 hours . Most stores
are open for business on Saturday and often on Sunday as well , except in a few places where prohibited by law ( see Blue law ) . Increasingly , employers are
offering compressed work schedules to employees . Some government and corporate employees now work a 9 / 80 work schedule ( 80 hours over 9 days during a two -
week period ) -- commonly 9 hour days Monday to Thursday , 8 hours on one Friday , and off the following Friday . Jobs in healthcare , law enforcement ,
transportation , retail , and other service positions commonly require employees to work on the weekend or to do shift work .
</item 1>

<item 2>
The workweek is time devoted to labour, contrasted with the weekend for rest. Monday–Friday work and Saturday–Sunday weekends predominate in the West, but
religious and regional customs vary: Israel generally rests Friday–Saturday, while Muslim-majority countries have shifted from Thursday/Friday or
Friday/Saturday toward Saturday/Sunday. The modern weekend emerged in 19th-century Britain; five-day weeks spread in the United States, becoming widespread
after the 1938 Fair Labor Standards Act established a
</item 2>

<item 3>
Workweek and weekend
A series of workweek reforms in the mid-to - late 2000s and early 2010s brought much of the Arab World in synchronization with the majority of countries around
the world , in terms of working hours , the length of the workweek , and the days of the weekend . The International Labour Organization ( ILO ) currently
defines a workweek exceeding 48 hours as excessive . A 2007 study by the ILO found that at least 614.2 million people around the world were working excessive
hours . Length ( edit ) This day planner chart ( which can be used for any months ) shows the workweek days as white boxes and the weekend days as light blue -
coloured boxes . Actual workweek lengths have been falling in the developed world . Every reduction of the length of the workweek has been accompanied by an
increase in real per - capita income . In the United States , the workweek length reduced slowly from before the Civil War to the turn of the 20th century . A
rapid reduction took place from 1900 to 1920 , especially between 1913 and 1919 , when weekly hours fell by about eight percent .
</item 3>

<item 4>
Workweek and weekend
In some Christian traditions , Sunday is the `` day of rest and worship '' . Jewish Shabbat or Biblical Sabbath lasts from sunset on Friday to the fall of full
darkness on Saturday ; as a result , the weekend in Israel is observed on Friday -- Saturday . Some Muslim - majority countries historically had a Thursday --
Friday or Friday -- Saturday weekend ; however , recently many such countries have shifted from Thursday -- Friday to Friday -- Saturday , or to Saturday --
Sunday . The Christian Sabbath was just one day each week , but the preceding day ( the Jewish Sabbath ) came to be taken as a holiday as well in the twentieth
century . This shift has been accompanied by a reduction in the total number of hours worked per week , following changes in employer expectations . The present
- day concept of the ' week - end ' first arose in the industrial north of Britain in the early part of nineteenth century and was originally a voluntary
arrangement between factory owners and workers allowing Saturday afternoon off from 2pm in agreement that staff would be available for work sober and refreshed
on Monday morning . The Amalgamated Clothing Workers of America Union was the first to successfully demand a five - day work week in 1929 .
</item 4>

<item 5>
Workweek and weekend
1 Soviet Union 3.17 Singapore 3.18 South Africa 3.19 Thailand 3.20 United States 3.21 Vietnam 4 See also 5 References History ( edit ) Part of a series on
Organised labour Labour movement ( show ) Timeline New Unionism Proletariat Social Movement Unionism Socialism Syndicalism Anarcho - syndicalism Labour rights (
hide ) Child labour Eight - hour day Five - day work week Annual leave Collective bargaining Minimum wage Occupational safety and health Trade unions ( show )
Trade unions by country Trade union federations International comparisons ITUC IWA WFTU Strike action ( show ) Chronological list of strikes General strike
Secondary action Sitdown strike Work - to - rule Labour parties ( show ) Australian British Dutch Irish Israeli New Zealand Norwegian List of Labour parties
Academic disciplines ( show ) Industrial relations Labour economics Labour history Labour law This section needs expansion . You can help by adding to it . (
September 2013 ) A continuous seven - day cycle that runs throughout history paying no attention whatsoever to the phases of the moon , having a fixed day of
rest , was probably first practiced in Judaism , dated to the 6th century BC at the latest .
</item 5>

<item 6>
Most countries have two-day weekends, but days vary by religious tradition; proposals for shorter workweeks cite social and economic benefits. Schedules differ
widely: China adopted Monday–Friday work in 1995, though many factories work Saturdays and holiday dates may be swapped; some regions encourage Friday-afternoon
starts. Hong Kong and Japan shifted many workers toward five-day weeks, while retail often operates weekends. Other examples include India’s common six-day
week, Mexico’s 48-hour week, and Russia’s
</item 6>

<item 7>
Workweek and weekend
( February 4 , 2010 ) `` Shorter Workweek in a Tough Economy '' Archived November 28 , 2016 , at the Wayback Machine . Wisconsin Employment Law Letter (
hrhero.com ) Jump up ^ Rones et al. ( 1997 ) `` Trends in Hours of Work since the Mid-1970s '' Archived June 24 , 2016 , at the Wayback Machine . Monthly Labor
Review 120 ( 3 ) : 3 -- 12 Jump up ^ Coote , Anna ; Franklin , Jane ; Simms , Andrew ( February 2010 ) . `` 21 hours : Why a shorter working week can help us
all to flourish in the 21st century '' ( PDF ) . London : New Economics Foundation . ISBN 978 - 1 - 904882 - 70 - 1 . Archived from the original ( PDF ) on 9
February 2016 . Retrieved 18 October 2016 . Jump up ^ Stuart , H.
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
A number of provinces and municipalities across China , including Hebei , Jiangxi and Chongqing , have issued new policies , calling on companies to create 2.5
- day weekends . Under the plan , government institutions , state - owned companies , joint - ventures and privately held companies are to be given incentives
to allow their workers to take off at noon on Friday before coming back to the office on Monday . Hong Kong ( edit ) In Hong Kong , a typical working week for
local enterprises begins on Monday and ends at 1pm on Saturday , although most employees have alternate Saturdays off . After the introduction of the five - day
working week for the majority of government departments in 2006 , most multinational enterprises and large local companies followed suit , extended the working
day from 9am to 6pm so as to adopt a five - day work week . Despite the aforementioned official hours , many employees work overtime , and in the case of the
financial industry in particular , working 12 - hour days on a chronic basis is not uncommon . Most commercial establishments in the retail sector such as
restaurants , shops and cinemas , as well as public venues such as museums and libraries are open on Saturdays , Sundays and most public holidays .
</item 9>

<item 10>
Workweek and weekend
In 1926 , Henry Ford standardized on a five - day workweek , instead of the prevalent six days , without reducing employees ' pay . Hours worked stabilized at
about 49 per week during the 1920s , and during the Great Depression fell below 40 . During the Depression , President Herbert Hoover called for a reduction in
work hours in lieu of layoffs . Later , President Franklin Roosevelt signed the Fair Labor Standards Act of 1938 , which established a five - day , 40 - hour
workweek for many workers . The proportion of people working very long weeks has since risen , and the full - time employment of women has increased
dramatically . The New Economics Foundation has recommended moving to a 21 - hour standard workweek to address problems with unemployment , high carbon
emissions , low well - being , entrenched inequalities , overworking , family care , and the general lack of free time . The Center for Economic and Policy
Research states that reducing the length of the work week would slow climate change and have other environmental benefits . Around the world ( edit ) (
Countries listed alphabetically . Some countries have Saturday a normal school day .
</item 10>
=== END ITEM id=39bcc1a56b302c98 ===

=== ITEM id=add55a3cda4332d7 ===
Gold sentences and facts:
Sentence 0: An elevator with a counterbalance approximates an ideal Atwood machine and thereby relieves the driving motor from the load of holding the elevator cab -- it has to overcome only weight difference and inertia of the two masses . The same principle is used for funicular railways with two connected railway cars on inclined tracks , and for the elevators on the Eiffel Tower which counterbalance each other . Ski lifts are another example , where the gondolas move on a closed ( continuous ) pulley system up and down the mountain . The ski lift is similar to the counter-weighted elevator , but with a constraining force provided by the cable in the vertical dimension thereby achieving work in both the horizontal and vertical dimensions . Boat lifts are another type of counter-weighted elevator system approximating an Atwood machine .
  Fact 0.0: An elevator with a counterbalance approximates an ideal Atwood machine.
  Fact 0.1: A counterbalanced elevator relieves the driving motor from the load of holding the elevator cab.
  Fact 0.2: A counterbalanced elevator's driving motor has to overcome only the weight difference and inertia of the two masses.
  Fact 0.3: The same principle is used for funicular railways with two connected railway cars on inclined tracks.
  Fact 0.4: The same principle is used for the elevators on the Eiffel Tower, which counterbalance each other.
  Fact 0.5: Ski lifts are another example.
  Fact 0.6: In ski lifts, the gondolas move on a closed (continuous) pulley system up and down the mountain.
  Fact 0.7: A ski lift is similar to the counter-weighted elevator, but with a constraining force provided by the cable in the vertical dimension.
  Fact 0.8: A ski lift achieves work in both the horizontal and vertical dimensions.
  Fact 0.9: Boat lifts are another type of counter-weighted elevator system approximating an Atwood machine.

Retrieved items:
<item 1>
Atwood machine
Atwood machine - wikipedia Atwood machine Jump to : navigation , search Illustration of Atwood machine , 1905 . The Atwood machine ( or Atwood 's machine ) was
invented in 1784 by the English mathematician George Atwood as a laboratory experiment to verify the mechanical laws of motion with constant acceleration .
Atwood 's machine is a common classroom demonstration used to illustrate principles of classical mechanics . The ideal Atwood Machine consists of two objects of
mass m and m , connected by an inextensible massless string over an ideal massless pulley . When m = m , the machine is in neutral equilibrium regardless of the
position of the weights . When m ≠ m both masses experience uniform acceleration . Contents ( hide ) 1 Equation for constant acceleration 2 Equation for tension
3 Equations for a pulley with inertia and friction 3.1 Practical implementations 4 See also 5 Notes 6 External links Equation for constant acceleration ( edit )
The free body diagrams of the two hanging masses of the Atwood machine .
</item 1>

<item 2>
Atwood machine
+ τ f r i c t i o n r g ) m 1 + m 2 + I r 2 ( \ displaystyle T_ ( 2 ) = ( ( m_ ( 2 ) g ( 2m_ ( 1 ) + ( ( I ) \ over ( r ^ ( 2 ) ) ) + ( ( \ tau _ ( \ mathrm (
friction ) ) ) \ over ( rg ) ) ) ) \ over ( m_ ( 1 ) + m_ ( 2 ) + ( ( I ) \ over ( r ^ ( 2 ) ) ) ) ) ) Should bearing friction be negligible ( but not the
inertia of the pulley and not the traction of the string on the pulley rim ) , these equations simplify as the following results : Acceleration : a = g ( m 1 −
m 2 ) m 1 + m 2 + I r 2 ( \ displaystyle a = ( ( g ( m_ ( 1 ) - m_ ( 2 ) ) ) \ over ( m_ ( 1 ) + m_ ( 2 ) + ( ( I ) \ over ( r ^ ( 2 ) ) ) ) ) )
</item 2>

<item 3>
Atwood machine
Weight of m 1 ( \ displaystyle m_ ( 1 ) ) and m 2 ( \ displaystyle m_ ( 2 ) ) is simply W 1 = m 1 g ( \ displaystyle W_ ( 1 ) = m_ ( 1 ) g ) and W 2 = m 2 g ( \
displaystyle W_ ( 2 ) = m_ ( 2 ) g ) respectively . Forces affecting m : m 1 g − T = m 1 a ( \ displaystyle \ ; m_ ( 1 ) g-T = m_ ( 1 ) a ) Forces affecting m :
T − m 2 g = m 2 a ( \ displaystyle \ ; T - m_ ( 2 ) g = m_ ( 2 ) a ) and adding the two previous equations we obtain m 1 g − m 2 g = m 1 a + m 2 a ( \
displaystyle \ ; m_ ( 1 ) g - m_ ( 2 ) g = m_ ( 1 ) a + m_ ( 2 ) a ) ,
</item 3>

<item 4>
Atwood machine
ISBN 81 - 85015 - 53 - 8 . Section 1 - 6 , example 2 Jump up ^ See Wikipedia article : https://en.wikipedia.org/wiki/Boat_lift External links ( edit ) Professor
Greenslade 's account on the Atwood Machine Atwood 's Machine by Enrique Zeleny , The Wolfram Demonstrations Project . Retrieved from ``
https://en.wikipedia.org/w/index.php?title=Atwood_machine&oldid=792136529 '' Categories : Mechanics Physics experiments Talk Contents About Wikipedia Català
Deutsch Español Français 한국어 Hrvatski Italiano Nederlands 日本 語 Polski Português Русский Srpskohrvatski / српскохрватски Türkçe Українська 中文 Edit links This
page was last edited on 24 July 2017 , at 17 : 58 . Text is available under the Creative Commons Attribution - ShareAlike License ; additional terms may apply .
By using this site , you agree to the Terms of Use and Privacy Policy .
</item 4>

<item 5>
Atwood machine
The angular acceleration of the pulley is given by the no - slip condition : α = a r , ( \ displaystyle \ alpha = ( a \ over r ) , ) where α ( \ displaystyle \
alpha ) is the angular acceleration . The net torque is then : τ n e t = ( T 1 − T 2 ) r − τ f r i c t i o n = I α ( \ displaystyle \ tau _ ( \ mathrm ( net ) )
= \ left ( T_ ( 1 ) - T_ ( 2 ) \ right ) r - \ tau _ ( \ mathrm ( friction ) ) = I \ alpha ) Combining with Newton 's second law for the hanging masses , and
solving for T , T , and a , we get : Acceleration : a = g ( m 1 − m 2 ) − τ f r i c t i o n r m 1 + m 2 + I r 2 ( \ displaystyle a = ( ( g ( m_ ( 1 ) - m_ ( 2 )
) - ( \ tau _ ( \ mathrm ( friction ) ) \ over r ) ) \ over ( m_ (
</item 5>

<item 6>
Atwood machine
Boat lifts are another type of counter-weighted elevator system approximating an Atwood machine . See also ( edit ) Wikimedia Commons has media related to
Atwood 's machine . Frictionless plane Kater 's pendulum Spherical cow Swinging Atwood 's machine Notes ( edit ) Jump up ^ Tipler , Paul A. ( 1991 ) . Physics
For Scientists and Engineers ( 3rd , extended ed . ) . New York : Worth Publishers . p. 160 . ISBN 0 - 87901 - 432 - 6 . Chapter 6 , example 6 - 13 Jump up ^
Goldstein , Herbert ( 1980 ) . Classical Mechanics ( 2nd ed . ) . New Delhi : Addison - Wesley / Narosa Indian Student Edition . pp. 26 -- 27 . ISBN 81 - 85015
- 53 - 8 .
</item 6>

<item 7>
Atwood machine
_ ( \ mathrm ( friction ) ) \ over r ) ) \ over ( m_ ( 1 ) + m_ ( 2 ) + ( ( I ) \ over ( r ^ ( 2 ) ) ) ) ) ) Tension in string segment nearest m : T 1 = m 1 g (
2 m 2 + I r 2 + τ f r i c t i o n r g ) m 1 + m 2 + I r 2 ( \ displaystyle T_ ( 1 ) = ( ( m_ ( 1 ) g ( 2m_ ( 2 ) + ( ( I ) \ over ( r ^ ( 2 ) ) ) + ( ( \ tau _
( \ mathrm ( friction ) ) ) \ over ( rg ) ) ) ) \ over ( m_ ( 1 ) + m_ ( 2 ) + ( ( I ) \ over ( r ^ ( 2 ) ) ) ) ) ) Tension in string segment nearest m : T 2 =
m 2 g ( 2 m 1 + I r 2 + τ f r i c t i o n r g ) m 1 + m 2
</item 7>

<item 8>
Atwood machine
( m_ ( 1 ) + m_ ( 2 ) + ( ( I ) \ over ( r ^ ( 2 ) ) ) ) ) ) Practical implementations ( edit ) Atwood 's original illustrations show the main pulley 's axle
resting on the rims of another four wheels , to minimize friction forces from the bearings . Many historical implementations of the machine follow this design .
An elevator with a counterbalance approximates an ideal Atwood machine and thereby relieves the driving motor from the load of holding the elevator cab -- it
has to overcome only weight difference and inertia of the two masses . The same principle is used for funicular railways with two connected railway cars on
inclined tracks , and for the elevators on the Eiffel Tower which counterbalance each other . Ski lifts are another example , where the gondolas move on a
closed ( continuous ) pulley system up and down the mountain . The ski lift is similar to the counter-weighted elevator , but with a constraining force provided
by the cable in the vertical dimension thereby achieving work in both the horizontal and vertical dimensions . Boat lifts are another type of counter-weighted
elevator system approximating an Atwood machine .
</item 8>

<item 9>
Atwood machine
By using this site , you agree to the Terms of Use and Privacy Policy . Wikipedia ® is a registered trademark of the Wikimedia Foundation , Inc. , a non-profit
organization . About Wikipedia
</item 9>

<item 10>
Atwood machine
) + ( ( I ) \ over ( r ^ ( 2 ) ) ) ) ) ) Tension in string segment nearest m : T 1 = m 1 g ( 2 m 2 + I r 2 ) m 1 + m 2 + I r 2 ( \ displaystyle T_ ( 1 ) = ( (
m_ ( 1 ) g ( 2m_ ( 2 ) + ( ( I ) \ over ( r ^ ( 2 ) ) ) ) ) \ over ( m_ ( 1 ) + m_ ( 2 ) + ( ( I ) \ over ( r ^ ( 2 ) ) ) ) ) ) Tension in string segment
nearest m : T 2 = m 2 g ( 2 m 1 + I r 2 ) m 1 + m 2 + I r 2 ( \ displaystyle T_ ( 2 ) = ( ( m_ ( 2 ) g ( 2m_ ( 1 ) + ( ( I ) \ over ( r ^ ( 2 ) ) ) ) ) \ over (
m_ ( 1 ) + m_ ( 2 ) + ( ( I ) \
</item 10>
=== END ITEM id=add55a3cda4332d7 ===

=== ITEM id=6a54c81251900406 ===
Gold sentences and facts:
Sentence 0: Year Winning team Losing team Score Site Notes 1909 Pittsburgh Pirates Detroit Tigers 8 -- 0 Bennett Park , Detroit Rookie Babe Adams tosses a six hit shutout for his third win of the series . 1912 Boston Red Sox New York Giants 3 -- 2 F / 10 Fenway Park , Boston This decisive contest was actually game eight , as game two had ended in a tie . 1924 Washington Senators New York Giants 4 -- 3 F / 12 Griffith Stadium , Washington , D.C. Longest game seven in innings . In the 12th , Muddy Ruel doubled after a dropped foul popfly by Giants catcher Hank Gowdy . Ruel scored on a game - winning RBI double by Earl McNeely . 1925 Pittsburgh Pirates Washington Senators 9 -- 7 Forbes Field , Pittsburgh Pittsburgh became the first team in a best of seven to win the series when trailing three games to one . 1926 St. Louis Cardinals New York Yankees 3 -- 2 Yankee Stadium , the Bronx , New York City Grover Cleveland Alexander strikes out Tony Lazzeri with the bases loaded to end the seventh and preserve the Cardinals lead . Two innings later Babe Ruth is caught stealing to end the World Series . 1931 St. Louis Cardinals Philadelphia Athletics 4 -- 2 St. Louis , Missouri Last postseason game for the Athletics while based in Philadelphia . 1934 St. Louis Cardinals Detroit Tigers 11 -- 0 Navin Field , Detroit Dizzy Dean 's second win , a six - hit shutout on one day of rest , gives the Cardinals the title . 1940 Cincinnati Reds Detroit Tigers 2 -- 1 Crosley Field , Cincinnati This game was over in a snappy 1 : 47 while a sacrifice bunt and sacrifice fly scored the winning runs . 1945 Detroit Tigers Chicago Cubs 9 -- 3 Wrigley Field , Chicago Cubs last World Series appearance until 2016 . See also : Curse of the Billy Goat 1946 St. Louis Cardinals Boston Red Sox 4 -- 3 Sportsman 's Park III , St. Louis , Missouri Enos Slaughter 's famous mad dash scored the deciding run to win the game and the series . 1947 New York Yankees Brooklyn Dodgers 5 -- 2 Yankee Stadium , the Bronx , New York City This was the Series that featured Cookie Lavagetto 's pinch - hit that spoiled Bill Bevens ' no - hit bid and Al Gionfriddo 's catch to rob Joe DiMaggio of a home run . 1952 New York Yankees Brooklyn Dodgers 4 -- 2 Ebbets Field , Brooklyn , New York City Billy Martin 's game saving infield catch preserves the Yankees lead and helps win the game and the series . 1955 Brooklyn Dodgers New York Yankees 2 -- 0 Yankee Stadium , the Bronx , New York City Dodgers ' only championship in Brooklyn . The home team won all games of the series , except for game seven . 1956 New York Yankees Brooklyn Dodgers 9 -- 0 Ebbets Field , Brooklyn , New York City Jackie Robinson 's final Major League game . Last postseason game at Ebbets Field and Dodgers ' last postseason game before moving to Los Angeles . The home team won all games of the series , except for game seven . 1957 Milwaukee Braves New York Yankees 5 -- 0 Yankee Stadium , the Bronx , New York City The Braves ' only championship in Milwaukee . 1958 New York Yankees Milwaukee Braves 6 -- 2 County Stadium , Milwaukee The Yankees win their 18th World Series title after being down three games to one . This was the Braves ' last postseason game in Milwaukee . 1960 Pittsburgh Pirates New York Yankees 10 -- 9 Forbes Field , Pittsburgh Bill Mazeroski hit the walk - off home run to win the Pirates the championship . Last postseason game in Forbes Field . Last postseason game for Yankees ' manager Casey Stengel . Highest scoring game seven . 1962 New York Yankees San Francisco Giants 1 -- 0 Candlestick Park , San Francisco With runners on second and third and two outs , Willie McCovey 's line drive to Bobby Richardson ended the game . First game seven that ended with a 1 - 0 score . 1964 St. Louis Cardinals New York Yankees 7 -- 5 Busch Stadium I , St. Louis , Missouri The Yankees ' last World Series appearance until 1976 . Last postseason game at Busch Stadium I . 1965 Los Angeles Dodgers Minnesota Twins 2 -- 0 Metropolitan Stadium , Bloomington , Minnesota Pitching through immense pain and on two days rest , Sandy Koufax throws a 3 hit shutout to win the Series and World Series MVP . The home team won all games of the series , except for game seven . 1967 St. Louis Cardinals Boston Red Sox 7 -- 2 Fenway Park , Boston Bob Gibson wins his third game of the Series and also homers . 1968 Detroit Tigers St. Louis Cardinals 4 -- 1 Busch Stadium II , St. Louis , Missouri Mickey Lolich out - duels Bob Gibson . The Tigers rallied from a 3 - 1 deficit to win the series . First game seven loss by the Cardinals in the World Series . 1971 Pittsburgh Pirates Baltimore Orioles 2 -- 1 Memorial Stadium , Baltimore The home team won all games of the series , except for game seven . 1972 Oakland Athletics Cincinnati Reds 3 -- 2 Riverfront Stadium , Cincinnati The A 's first championship in Oakland . The A 's were the last American League team to win a World Series game seven on the road until 2017 . 1973 Oakland Athletics New York Mets 5 -- 2 Oakland -- Alameda County Coliseum , Oakland , California Bert Campaneris and Reggie Jackson hit the only Oakland home runs in this game to set the tone for victory and the series . Cincinnati Reds Boston Red Sox 4 -- 3 Fenway Park , Boston One night after Carlton Fisk 's famous home run , the Reds rebound to take the Series . 1979 Pittsburgh Pirates Baltimore Orioles 4 -- 1 Memorial Stadium , Baltimore The Pittsburgh Pirates , powered by NLCS and World Series MVP Willie Stargell , rally from a 3 - 1 deficit to win the series . The Pirates were the last team to win a road game seven until 2014 . 1982 St. Louis Cardinals Milwaukee Brewers 6 -- 3 Busch Stadium II , St. Louis , Missouri Joaquín Andújar earns his second win of the Series and Bruce Sutter his second save . 1985 Kansas City Royals St. Louis Cardinals 11 -- 0 Royals Stadium , Kansas City , Missouri Second World Series held entirely in the state of Missouri . The Royals also won the 1985 American League Championship Series in seven games . Royals became first team to win the World Series after losing the first two games at home . The Royals were the last team to come back from 3 -- 1 down to win a title , in any of the five major men 's professional sports leagues , until the Cleveland Cavaliers won the 2016 NBA Finals after trailing 3 -- 1 . The 1985 Royals were the last team to win a World Series after trailing 3 -- 1 until the Cubs did so in 2016 . 1986 New York Mets Boston Red Sox 8 -- 5 Shea Stadium , Flushing , Queens , New York Mets become the first team to be one strike away from elimination and come back to win the series . They also become the second team to win the series after losing the first two at home . Minnesota Twins St. Louis Cardinals 4 -- 2 Hubert H. Humphrey Metrodome , Minneapolis The home team won all seven games of the series for the first time in MLB history . 1991 Minnesota Twins Atlanta Braves 1 -- 0 F / 10 Hubert H. Humphrey Metrodome , Minneapolis Like in 1987 , the home team won all games of the series . The Twins won in the bottom of the 10th on a walk - off single by Gene Larkin scoring Dan Gladden after ​ 9 ⁄ scoreless innings of play . Jack Morris goes all ten innings for the Twins and is named the Series MVP . Twins Centerfielder Kirby Puckett hit a famous walk - off home run in the bottom of the eleventh inning of the previous night 's game to force this decisive game . 1997 Florida Marlins Cleveland Indians 3 -- 2 F / 11 Pro Player Stadium , Miami Gardens , Florida Florida becomes the first wild card team to win the World Series . The Marlins rallied for one run in the bottom of the ninth to tie and won in the eleventh on an Édgar Rentería RBI single off Charles Nagy . 2001 Arizona Diamondbacks New York Yankees 3 -- 2 Bank One Ballpark , Phoenix , Arizona Arizona becomes the fastest expansion team to win the World Series , rallying for two runs in the bottom of the ninth for a come - from - behind win , capped off by Luis Gonzalez hitting the winning RBI single off closer Mariano Rivera . The home team won every game of the series . 2002 Anaheim Angels San Francisco Giants 4 -- 1 Edison International Field , Anaheim , California Both World Series teams were wild card entrants into the playoffs . John Lackey became the second rookie pitcher to win a World Series game seven . 2011 St. Louis Cardinals Texas Rangers 6 -- 2 Busch Stadium , St. Louis , Missouri Cardinals became the second team to be one strike away from elimination and come back to win the series , which they did twice . Just like the 1991 series , a walk - off home run in the bottom of the eleventh inning of game six , forced this decisive game with David Freese providing the fire power . Most recent game seven won by the home team . 2014 San Francisco Giants Kansas City Royals 3 -- 2 Kauffman Stadium , Kansas City , Missouri Like in 2002 , both World Series teams were wild card entrants into the playoffs . The Giants became the first NL team to win a game seven on the road since 1979 . NLCS MVP and eventual World Series MVP Madison Bumgarner , normally a starter , pitched five scoreless innings of relief on two days rest to save the Giants ' third title in five years . 2016 Chicago Cubs Cleveland Indians 8 -- 7 F / 10 Progressive Field , Cleveland , Ohio Chicago Cubs ' first World Series win since 1908 . Game played between the two teams ( Cleveland Indians and Chicago Cubs ) with the longest active championship droughts . The Cubs are the first Major League Baseball team to come back to win the World Series from a 3 -- 1 series deficit since the 1985 Royals . Most recent World Series game seven to be decided in extra innings . First extra innings game seven to be won by the road team . 2017 Houston Astros Los Angeles Dodgers 5 -- 1 Dodger Stadium , Los Angeles First instance of back - to - back game sevens since 2001 -- 2002 . First World Series championship for the Astros , who became the first AL team to win a World Series game 7 on the road since 1972 . The Astros became the second team to win two winner - take - all game sevens in the same postseason since the Kansas City Royals in 1985 .
  Fact 0.0: 1909 World Series: Pittsburgh Pirates beat Detroit Tigers 8-0 at Bennett Park, Detroit.
  Fact 0.1: 1909 World Series: Rookie Babe Adams tossed a six-hit shutout for his third win of the series.
  Fact 0.2: 1912 World Series: Boston Red Sox beat New York Giants 3-2 in 10 innings (F/10) at Fenway Park, Boston.
  Fact 0.3: 1912 World Series: The decisive contest was actually game eight, as game two had ended in a tie.
  Fact 0.4: 1924 World Series: Washington Senators beat New York Giants 4-3 in 12 innings (F/12) at Griffith Stadium, Washington, D.C.
  Fact 0.5: 1924 World Series: It was the longest game seven in innings.
  Fact 0.6: 1924 World Series: In the 12th inning, Muddy Ruel doubled after a dropped foul popfly by Giants catcher Hank Gowdy.
  Fact 0.7: 1924 World Series: Muddy Ruel scored on a game-winning RBI double by Earl McNeely.
  Fact 0.8: 1925 World Series: Pittsburgh Pirates beat Washington Senators 9-7 at Forbes Field, Pittsburgh.
  Fact 0.9: 1925 World Series: Pittsburgh became the first team in a best-of-seven series to win when trailing three games to one.
  Fact 0.10: 1926 World Series: St. Louis Cardinals beat New York Yankees 3-2 at Yankee Stadium, the Bronx, New York City.
  Fact 0.11: 1926 World Series: Grover Cleveland Alexander struck out Tony Lazzeri with the bases loaded to end the seventh inning and preserve the Cardinals lead.
  Fact 0.12: 1926 World Series: Two innings later, Babe Ruth was caught stealing to end the World Series.
  Fact 0.13: 1931 World Series: St. Louis Cardinals beat Philadelphia Athletics 4-2 at St. Louis, Missouri.
  Fact 0.14: 1931 World Series: It was the last postseason game for the Athletics while based in Philadelphia.
  Fact 0.15: 1934 World Series: St. Louis Cardinals beat Detroit Tigers 11-0 at Navin Field, Detroit.
  Fact 0.16: 1934 World Series: Dizzy Dean's second win, a six-hit shutout on one day of rest, gave the Cardinals the title.
  Fact 0.17: 1940 World Series: Cincinnati Reds beat Detroit Tigers 2-1 at Crosley Field, Cincinnati.
  Fact 0.18: 1940 World Series: The game was over in a snappy 1:47.
  Fact 0.19: 1940 World Series: A sacrifice bunt and a sacrifice fly scored the winning runs.
  Fact 0.20: 1945 World Series: Detroit Tigers beat Chicago Cubs 9-3 at Wrigley Field, Chicago.
  Fact 0.21: 1945 World Series: It was the Cubs' last World Series appearance until 2016.
  Fact 0.22: 1945 World Series: The row has a see-also note for the Curse of the Billy Goat.
  Fact 0.23: 1946 World Series: St. Louis Cardinals beat Boston Red Sox 4-3 at Sportsman's Park III, St. Louis, Missouri.
  Fact 0.24: 1946 World Series: Enos Slaughter's famous mad dash scored the deciding run to win the game and the series.
  Fact 0.25: 1947 World Series: New York Yankees beat Brooklyn Dodgers 5-2 at Yankee Stadium, the Bronx, New York City.
  Fact 0.26: 1947 World Series: This was the Series that featured Cookie Lavagetto's pinch-hit that spoiled Bill Bevens' no-hit bid.
  Fact 0.27: 1947 World Series: Al Gionfriddo's catch robbed Joe DiMaggio of a home run.
  Fact 0.28: 1952 World Series: New York Yankees beat Brooklyn Dodgers 4-2 at Ebbets Field, Brooklyn, New York City.
  Fact 0.29: 1952 World Series: Billy Martin's game-saving infield catch preserved the Yankees lead and helped win the game and the series.
  Fact 0.30: 1955 World Series: Brooklyn Dodgers beat New York Yankees 2-0 at Yankee Stadium, the Bronx, New York City.
  Fact 0.31: 1955 World Series: It was the Dodgers' only championship in Brooklyn.
  Fact 0.32: 1955 World Series: The home team won all games of the series except game seven.
  Fact 0.33: 1956 World Series: New York Yankees beat Brooklyn Dodgers 9-0 at Ebbets Field, Brooklyn, New York City.
  Fact 0.34: 1956 World Series: It was Jackie Robinson's final Major League game.
  Fact 0.35: 1956 World Series: It was the last postseason game at Ebbets Field.
  Fact 0.36: 1956 World Series: It was the Dodgers' last postseason game before moving to Los Angeles.
  Fact 0.37: 1956 World Series: The home team won all games of the series except game seven.
  Fact 0.38: 1957 World Series: Milwaukee Braves beat New York Yankees 5-0 at Yankee Stadium, the Bronx, New York City.
  Fact 0.39: 1957 World Series: It was the Braves' only championship in Milwaukee.
  Fact 0.40: 1958 World Series: New York Yankees beat Milwaukee Braves 6-2 at County Stadium, Milwaukee.
  Fact 0.41: 1958 World Series: The Yankees won their 18th World Series title after being down three games to one.
  Fact 0.42: 1958 World Series: It was the Braves' last postseason game in Milwaukee.
  Fact 0.43: 1960 World Series: Pittsburgh Pirates beat New York Yankees 10-9 at Forbes Field, Pittsburgh.
  Fact 0.44: 1960 World Series: Bill Mazeroski hit the walk-off home run to win the Pirates the championship.
  Fact 0.45: 1960 World Series: It was the last postseason game in Forbes Field.
  Fact 0.46: 1960 World Series: It was the last postseason game for Yankees' manager Casey Stengel.
  Fact 0.47: 1960 World Series: It was the highest scoring game seven.
  Fact 0.48: 1962 World Series: New York Yankees beat San Francisco Giants 1-0 at Candlestick Park, San Francisco.
  Fact 0.49: 1962 World Series: With runners on second and third and two outs, Willie McCovey's line drive to Bobby Richardson ended the game.
  Fact 0.50: 1962 World Series: It was the first game seven that ended with a 1-0 score.
  Fact 0.51: 1964 World Series: St. Louis Cardinals beat New York Yankees 7-5 at Busch Stadium I, St. Louis, Missouri.
  Fact 0.52: 1964 World Series: It was the Yankees' last World Series appearance until 1976.
  Fact 0.53: 1964 World Series: It was the last postseason game at Busch Stadium I.
  Fact 0.54: 1965 World Series: Los Angeles Dodgers beat Minnesota Twins 2-0 at Metropolitan Stadium, Bloomington, Minnesota.
  Fact 0.55: 1965 World Series: Sandy Koufax pitched through immense pain and on two days rest.
  Fact 0.56: 1965 World Series: Sandy Koufax threw a 3-hit shutout to win the Series.
  Fact 0.57: 1965 World Series: Sandy Koufax was named World Series MVP.
  Fact 0.58: 1965 World Series: The home team won all games of the series except game seven.
  Fact 0.59: 1967 World Series: St. Louis Cardinals beat Boston Red Sox 7-2 at Fenway Park, Boston.
  Fact 0.60: 1967 World Series: Bob Gibson wins his third game of the Series.
  Fact 0.61: 1967 World Series: Bob Gibson also homers in this game.
  Fact 0.62: 1968 World Series: Detroit Tigers beat St. Louis Cardinals 4-1 at Busch Stadium II, St. Louis, Missouri.
  Fact 0.63: 1968 World Series: Mickey Lolich out-dueled Bob Gibson.
  Fact 0.64: 1968 World Series: The Tigers rallied from a 3-1 deficit to win the series.
  Fact 0.65: 1968 World Series: It was the first game seven loss by the Cardinals in the World Series.
  Fact 0.66: 1971 World Series: Pittsburgh Pirates beat Baltimore Orioles 2-1 at Memorial Stadium, Baltimore.
  Fact 0.67: 1971 World Series: The home team won all games of the series except game seven.
  Fact 0.68: 1972 World Series: Oakland Athletics beat Cincinnati Reds 3-2 at Riverfront Stadium, Cincinnati.
  Fact 0.69: 1972 World Series: It was the A's first championship in Oakland.
  Fact 0.70: 1972 World Series: The A's were the last American League team to win a World Series game seven on the road until 2017.
  Fact 0.71: 1973 World Series: Oakland Athletics beat New York Mets 5-2 at Oakland-Alameda County Coliseum, Oakland, California.
  Fact 0.72: 1973 World Series: Bert Campaneris and Reggie Jackson hit the only Oakland home runs in this game.
  Fact 0.73: 1973 World Series: Those home runs set the tone for victory and the series.
  Fact 0.74: A row with no year listed: Cincinnati Reds beat Boston Red Sox 4-3 at Fenway Park, Boston.
  Fact 0.75: A row with no year listed: It was one night after Carlton Fisk's famous home run.
  Fact 0.76: A row with no year listed: The Reds rebounded to take the Series.
  Fact 0.77: 1979 World Series: Pittsburgh Pirates beat Baltimore Orioles 4-1 at Memorial Stadium, Baltimore.
  Fact 0.78: 1979 World Series: The Pirates were powered by NLCS and World Series MVP Willie Stargell.
  Fact 0.79: 1979 World Series: The Pirates rallied from a 3-1 deficit to win the series.
  Fact 0.80: 1979 World Series: The Pirates were the last team to win a road game seven until 2014.
  Fact 0.81: 1982 World Series: St. Louis Cardinals beat Milwaukee Brewers 6-3 at Busch Stadium II, St. Louis, Missouri.
  Fact 0.82: 1982 World Series: Joaquín Andújar earned his second win of the Series.
  Fact 0.83: 1982 World Series: Bruce Sutter earned his second save.
  Fact 0.84: 1985 World Series: Kansas City Royals beat St. Louis Cardinals 11-0 at Royals Stadium, Kansas City, Missouri.
  Fact 0.85: 1985 World Series: It was the second World Series held entirely in the state of Missouri.
  Fact 0.86: 1985 Royals also won the 1985 American League Championship Series in seven games.
  Fact 0.87: 1985 Royals became the first team to win the World Series after losing the first two games at home.
  Fact 0.88: 1985 Royals were the last team to come back from 3-1 down to win a title in any of the five major men's professional sports leagues until the Cleveland Cavaliers won the 2016 NBA Finals after trailing 3-1.
  Fact 0.89: 1985 Royals were the last team to win a World Series after trailing 3-1 until the Cubs did so in 2016.
  Fact 0.90: 1986 World Series: New York Mets beat Boston Red Sox 8-5 at Shea Stadium, Flushing, Queens, New York.
  Fact 0.91: 1986 World Series: The Mets became the first team to be one strike away from elimination and come back to win the series.
  Fact 0.92: 1986 World Series: The Mets became the second team to win the series after losing the first two games at home.
  Fact 0.93: A row with no year listed: Minnesota Twins beat St. Louis Cardinals 4-2 at Hubert H. Humphrey Metrodome, Minneapolis.
  Fact 0.94: A row with no year listed: The home team won all seven games of the series for the first time in MLB history.
  Fact 0.95: 1991 World Series: Minnesota Twins beat Atlanta Braves 1-0 in 10 innings (F/10) at Hubert H. Humphrey Metrodome, Minneapolis.
  Fact 0.96: 1991 World Series: Like in 1987, the home team won all games of the series.
  Fact 0.97: 1991 World Series: The Twins won in the bottom of the 10th on a walk-off single by Gene Larkin scoring Dan Gladden after 9 and a fraction scoreless innings of play.
  Fact 0.98: 1991 World Series: Jack Morris went all ten innings for the Twins and was named Series MVP.
  Fact 0.99: 1991 World Series: Twins centerfielder Kirby Puckett hit a famous walk-off home run in the bottom of the eleventh inning of the previous night's game to force this decisive game.
  Fact 0.100: 1997 World Series: Florida Marlins beat Cleveland Indians 3-2 in 11 innings (F/11) at Pro Player Stadium, Miami Gardens, Florida.
  Fact 0.101: 1997 World Series: Florida became the first wild card team to win the World Series.
  Fact 0.102: 1997 World Series: The Marlins rallied for one run in the bottom of the ninth to tie.
  Fact 0.103: 1997 World Series: The Marlins won in the eleventh on an Édgar Rentería RBI single off Charles Nagy.
  Fact 0.104: 2001 World Series: Arizona Diamondbacks beat New York Yankees 3-2 at Bank One Ballpark, Phoenix, Arizona.
  Fact 0.105: 2001 World Series: Arizona became the fastest expansion team to win the World Series.
  Fact 0.106: 2001 World Series: Arizona rallied for two runs in the bottom of the ninth for a come-from-behind win.
  Fact 0.107: 2001 World Series: Luis Gonzalez hit the winning RBI single off closer Mariano Rivera.
  Fact 0.108: 2001 World Series: The home team won every game of the series.
  Fact 0.109: 2002 World Series: Anaheim Angels beat San Francisco Giants 4-1 at Edison International Field, Anaheim, California.
  Fact 0.110: 2002 World Series: Both World Series teams were wild card entrants into the playoffs.
  Fact 0.111: 2002 World Series: John Lackey became the second rookie pitcher to win a World Series game seven.
  Fact 0.112: 2011 World Series: St. Louis Cardinals beat Texas Rangers 6-2 at Busch Stadium, St. Louis, Missouri.
  Fact 0.113: 2011 World Series: The Cardinals became the second team to be one strike away from elimination and come back to win the series.
  Fact 0.114: 2011 World Series: The Cardinals did come back to win the series twice.
  Fact 0.115: 2011 World Series: Just like the 1991 series, a walk-off home run in the bottom of the eleventh inning of game six forced this decisive game.
  Fact 0.116: 2011 World Series: David Freese provided the firepower in that game six walk-off home run.
  Fact 0.117: 2011 World Series: The most recent game seven was won by the home team.
  Fact 0.118: 2014 World Series: San Francisco Giants beat Kansas City Royals 3-2 at Kauffman Stadium, Kansas City, Missouri.
  Fact 0.119: 2014 World Series: Like in 2002, both World Series teams were wild card entrants into the playoffs.
  Fact 0.120: 2014 World Series: The Giants became the first NL team to win a game seven on the road since 1979.
  Fact 0.121: 2014 World Series: Madison Bumgarner was the NLCS MVP and eventual World Series MVP.
  Fact 0.122: 2014 World Series: Madison Bumgarner is normally a starter.
  Fact 0.123: 2014 World Series: Madison Bumgarner pitched five scoreless innings of relief on two days rest.
  Fact 0.124: 2014 World Series: Madison Bumgarner's relief pitching saved the Giants' third title in five years.
  Fact 0.125: 2016 World Series: Chicago Cubs beat Cleveland Indians 8-7 in 10 innings (F/10) at Progressive Field, Cleveland, Ohio.
  Fact 0.126: 2016 World Series: It was the Cubs' first World Series win since 1908.
  Fact 0.127: 2016 World Series: The game was played between the two teams with the longest active championship droughts.
  Fact 0.128: 2016 World Series: The Cubs are the first Major League Baseball team to come back to win the World Series from a 3-1 series deficit since the 1985 Royals.
  Fact 0.129: 2016 World Series: It was the most recent World Series game seven to be decided in extra innings.
  Fact 0.130: 2016 World Series: It was the first extra innings game seven to be won by the road team.
  Fact 0.131: 2017 World Series: Houston Astros beat Los Angeles Dodgers 5-1 at Dodger Stadium, Los Angeles.
  Fact 0.132: 2017 World Series: It was the first instance of back-to-back game sevens since 2001-2002.
  Fact 0.133: 2017 World Series: It was the first World Series championship for the Astros.
  Fact 0.134: 2017 World Series: The Astros became the first AL team to win a World Series game 7 on the road since 1972.
  Fact 0.135: 2017 World Series: The Astros became the second team to win two winner-take-all game sevens in the same postseason since the 1985 Kansas City Royals.

Retrieved items:
<item 1>
Game seven
Game played between the two teams ( Cleveland Indians and Chicago Cubs ) with the longest active championship droughts . The Cubs are the first Major League
Baseball team to come back to win the World Series from a 3 -- 1 series deficit since the 1985 Royals . Most recent World Series game seven to be decided in
extra innings . First extra innings game seven to be won by the road team . 2017 Houston Astros Los Angeles Dodgers 5 -- 1 Dodger Stadium , Los Angeles First
instance of back - to - back game sevens since 2001 -- 2002 . First World Series championship for the Astros , who became the first AL team to win a World
Series game 7 on the road since 1972 . The Astros became the second team to win two winner - take - all game sevens in the same postseason since the Kansas City
Royals in 1985 . League Championship Series ( edit ) The League Championship Series were originally decided best of five from 1969 until 1984 . Since 1985 when
both were expanded to best of seven contests , 16 series ( 7 American , 9 National ) were decided in game seven .
</item 1>

<item 2>
Game seven
2012 NLCS San Francisco Giants St. Louis Cardinals 9 -- 0 AT&T Park , San Francisco , California First Giants victory in a winner - take - all game seven . The
Giants would go on to sweep the Detroit Tigers in the World Series . 2017 ALCS Houston Astros New York Yankees 4 -- 0 Minute Maid Park , Houston , Texas The
home team won all games of the series . Houston became the first franchise to win the pennant in both the National League and American League . The Astros would
go on to win the World Series over the Los Angeles Dodgers in seven games , becoming the second MLB team to win two game sevens in a single postseason .
National Hockey League ( edit ) Main article : List of NHL game sevens In the NHL playoffs , game seven can occur in all playoff series . In the Stanley Cup
Final , the NHL gained complete control of the Cup in 1926 , and the league instituted the best - of - seven series starting in the 1938 -- 39 season . No
Stanley Cup Final game seven has ever ended with a 1 -- 0 score .
</item 2>

<item 3>
Game seven
2003 NLCS Florida Marlins Chicago Cubs 9 -- 6 Wrigley Field , Chicago One night after the Steve Bartman incident , the Marlins advance to the World Series ,
where they would win in six games , and the Cubs ' World Series drought continues . ALCS Boston Red Sox New York Yankees 10 -- 3 Yankee Stadium , the Bronx ,
New York City Boston rallied from an 0 -- 3 deficit to win the American League pennant , the only time in Major League Baseball history that this has happened
in a best of seven series . Boston would go on to win the World Series for the first time since 1918 . NLCS St. Louis Cardinals Houston Astros 5 -- 2 Busch
Stadium ( II ) , St. Louis , Missouri The home team won all games of the series . Last time both LCS went to a game seven in the same year . Scott Rolen 's two
- run homer off Roger Clemens in the sixth inning puts the Cardinals in front for good . This is the last win of the season for St. Louis as the Cardinals got
steamrolled by Boston in four games . 2006 NLCS St.
</item 3>

<item 4>
2017 World Series
1997 1998 1999 2000s -- 2010s 2000 2001 2002 2003 2005 2006 2007 2008 2009 2011 2012 2013 2014 2015 2016 2017 See also Pre-World Series champions World Series
champions Most Valuable Players Starting pitchers Babe Ruth Award Commissioner 's Trophy World Series ring Appearances Streaks Droughts Series ( by franchise )
Broadcasters TV ratings ALCS NLCS ALDS NLDS ALWC NLWC Game 7 1989 Loma Prieta earthquake Dauvray Cup Temple Cup Chronicle - Telegraph Cup Book : World Series
Category : World Series Fox Major League Baseball Related programs The Cheap Seats ( 2010 -- 2011 ) MLB Whiparound Major League Baseball Game of the Week
Thursday Night Baseball ( 1997 -- 2001 ) This Week in Baseball ( 2000 -- 2011 ) Related articles DirecTV N3D FoxBox FoxTrax Scooter Television contracts ( cable
) MLB Network World Series television ratings National coverage Fox ( 1996 -- present ) FS1 ( 2014 -- present
</item 4>

<item 5>
Game seven
1 World Series 1.1. 2 League Championship Series 1.2 National Hockey League 1.3 National Basketball Association 1.4 Chinese Professional Baseball League 2
Comebacks 3 References Examples ( edit ) Major League Baseball ( edit ) In the MLB Postseason , game seven can occur in the League Championship Series and the
World Series ( the divisional series are best of five ) . In the World Series , there have been 39 decisive game sevens ( of the four World Series that were
contested on a best - of - nine basis , none went to a game nine , though all four included a non-decisive game seven ) . World Series ( edit ) World Series
that were decided on a game seven : All extra inning game sevens involve three franchises each with two appearances , the New York / San Francisco Giants (
losing both ) , the Washington Senators / Minnesota Twins ( winning both ) and the Cleveland Indians ( losing both ) . Home teams are 19 -- 20 in game sevens .
</item 5>

<item 6>
2005 World Series
Game Ratings ( households ) Share ( households ) American audience ( in millions ) 9.5 17 15.0 11.1 17 17.19 11.0 21 16.65 13.0 21 19.98 Aftermath ( edit )
Neither team advanced to the post-season in 2006 , but the 2006 World Series again featured teams from the American League Central and National League Central
divisions , this time represented by the Detroit Tigers and the St. Louis Cardinals , respectively . The Cardinals won the World Series in five games , in which
manager Tony La Russa became the second manager to win the World Series in both American and National leagues , previously managing the Oakland Athletics to the
1989 World Series championship . Both the White Sox and the Astros were in the Wild Card race until the final weeks of the season , with the White Sox finishing
with 90 wins , the Astros with 82 wins .
</item 6>

<item 7>
Game seven
1966 1967 1968 1969 1970 1971 1972 1973 1976 1977 1978 1979 1980s -- 1990s 1980 1981 1982 1984 1985 1986 1988 1989 1990 1991 1992 1993 1994 1995 1997 1998 1999
2000s -- 2010s 2000 2001 2002 2003 2005 2006 2007 2008 2009 2011 2012 2013 2014 2015 2016 2017 See also Pre-World Series champions World Series champions Most
Valuable Players Starting pitchers Babe Ruth Award Commissioner 's Trophy World Series ring Appearances Streaks Droughts Series ( by franchise ) Broadcasters TV
ratings ALCS NLCS ALDS NLDS ALWC NLWC Game 7 1989 Loma Prieta earthquake Dauvray Cup Temple Cup Chronicle - Telegraph Cup Book : World Series Category : World
Series Stanley Cup Finals Stanley Cup awarded on challenge basis 1893 --
</item 7>

<item 8>
2017 Stanley Cup playoffs
The Columbus Blue Jackets earned 108 points to finish third in the Metropolitan . This was the second playoff meeting between these teams ; they last met in the
2014 Eastern Conference First Round , which Pittsburgh won in six games . The teams split this year 's four - game regular season series . The Penguins defeated
the Blue Jackets in five games . Before game one began , Matt Murray , the starting goaltender for the Penguins , injured himself during warm - ups forcing Marc
- Andre Fleury , who had been demoted to the backup for the season , to start the game ; his team would win the game 3 -- 1 , making 31 saves in the process .
Sidney Crosby got a goal and two assists in game two as the Penguins defeated the Blue Jackets 4 -- 1 and took a 2 -- 0 series lead . In game three , Jake
Guentzel scored a hat - trick , his third being the overtime - winner as the Penguins came back from a two - goal deficit and defeated the Blue Jackets 5 -- 4
and take a 3 -- 0 series lead .
</item 8>

<item 9>
NBC Sunday Night Football
World Series conflicts ( edit ) In its first four seasons of Sunday night coverage , NBC took one week off in late October or early November , so as not to
conflict with Fox 's coverage of baseball 's World Series . In 2006 NBC did not air a game on October 22 , which was the scheduled date for Game 2 of the World
Series , but a potential conflict still existed on October 29 had the series gone seven games ( the conflict never arose , however , as the 2006 World Series
ended in five games ) . With the change in World Series scheduling beginning in 2007 , NBC did not air an NFL game in order to avoid a conflict with World
Series Game 4 , which is the first chance a team would have to clinch the series . In 2007 , there was no game on October 28 ; in 2008 , there was no game on
October 26 ; and , in 2009 , there was no game on November 1 . Although no games aired on these nights , Football Night in America still aired as scheduled at 7
: 00 p.m. Eastern .
</item 9>

<item 10>
2017 World Series
2018 World Series 1998 1999 2000 2001 2002 2003 2005 2006 2007 2008 2009 2011 2012 2013 2014 2015 2016 2017 2018 Related programs ESPN Major League Baseball
Major League Baseball Game of the Week Sunday Night Baseball Commentators All - Star Game ALCS ALDS NLCS NLDS World Series Lore 1998 Major League Baseball home
run record chase Death of Osama bin Laden Grand Slam Single Tie - breaker games 1998 NL Wild Card Playoff 1999 NL Wild Card Playoff 2007 NL Wild Card Playoff
2008 AL Central Playoff 2009 AL Central Playoff 2013 AL Wild Card tie - breaker Divisional Series games The Flip Play `` The Bug Game '' Roy Halladay 's
postseason no - hitter `` The Illegal Slide Game '' `` The 53 - Minute 7th Inning '' World Series games The Last Night of the Yankee Dynasty Game 6 of the 2011
World Series Walk - off Obstruction Chicago Cubs Steve Bartman ( 2003 ) Curse of the Billy Goat New York Yankees Yankees -- Red Sox rivalry Curse of the
</item 10>
=== END ITEM id=6a54c81251900406 ===

=== ITEM id=c27057603aad17d9 ===
Gold sentences and facts:
Sentence 0: The Eagles and the Patriots met again in Super Bowl LII , following the 2017 season , with the Eagles taking their revenge 41 -- 33 .
  Fact 0.0: The Eagles and the Patriots met again in Super Bowl LII.
  Fact 0.1: Super Bowl LII followed the 2017 season.
  Fact 0.2: The Eagles took their revenge in Super Bowl LII.
  Fact 0.3: The Eagles won Super Bowl LII 41 to 33.

Retrieved items:
<item 1>
Super Bowl XXXIX
The Eagles failed to recover their ensuing onside kick attempt , with Christian Fauria catching the ball cleanly and sliding down to seal possession for New
England . The Patriots then played it safe by running the ball 3 times and forcing Philadelphia to use all of its timeouts . New England punter Josh Miller then
pinned the Eagles back at their own 4 - yard line with just 46 seconds left in the game . Philadelphia then tried one last desperate drive to win or tie the
game . But on first down , McNabb was pressured into making a rushed pass to Westbrook at the line of scrimmage . Instead of dropping the pass to stop the clock
, Westbrook made the mistake of catching the ball and was immediately tackled for no gain , keeping the clock running and forcing the Eagles to run back to the
line of scrimmage for their next play with no huddle . On second down , McNabb threw an incomplete pass intended for Owens . Finally on third down , McNabb
threw a pass that deflected off of the outstretched fingertips of Smith and into the arms of Harrison for an interception with nine seconds left .
</item 1>

<item 2>
Super Bowl XXXIX was played February 6, 2005, at Alltel Stadium in Jacksonville, Florida. The New England Patriots defeated the Philadelphia Eagles 24–21,
winning their third championship in four seasons. Patriots wide receiver Deion Branch was named MVP. Attendance was 78,125, and Terry McAulay was the referee.
The Patriots were favored by seven points.
</item 2>

<item 3>
Super Bowl XXXIX
Super Bowl XXXIX - wikipedia Super Bowl XXXIX Jump to : navigation , search Super Bowl XXXIX New England Patriots ( AFC ) Philadelphia Eagles ( NFC ) 24 21
Total NE 0 7 7 10 24 PHI 0 7 7 7 21 Date February 6 , 2005 ( 2005 - 02 - 06 ) Stadium Alltel Stadium , Jacksonville , Florida MVP Deion Branch , Wide Receiver
Favorite Patriots by 7 Referee Terry McAulay Attendance 78,125 Future Hall of Famers Patriots : None Eagles : Brian Dawkins , Terrell Owens Ceremonies National
anthem The combined choirs of the U.S. Military Academy , the U.S. Naval Academy , the U.S. Air Force Academy , and the U.S. Coast Guard Academy , and U.S. Army
Herald Trumpets .
</item 3>

<item 4>
Philadelphia Eagles
Foles had his best game since week 15 and threw for 352 passing yards and 3 touchdowns including an impressive flea flicker touchdown pass . This win marked the
franchise 's third Super Bowl appearance and a berth in Super Bowl LII for a 2005 rematch against Tom Brady and the New England Patriots . With Foles at the
helm , the Eagles started off the game strong , leading the Patriots 22 -- 12 at halftime . In the second half the Patriots gained momentum and for the first
time took the lead in the 4th quarter 33 -- 32 . The Eagles rallied back and scored an 11 - yard Touchdown to tight end Zach Ertz . Later the Eagles converted
on a 46 - yard field goal by Jake Elliott to lead 41 -- 33 . On the final play of the game with 9 seconds left , Brady threw a hail - mary pass directed toward
tight end Rob Gronkowski which was bobbled by Patrick Robinson before being dropped in the end - zone as time expired , securing the win for the Eagles 41 -- 33
, and ending the 58 - year long championship drought .
</item 4>

<item 5>
Philadelphia Eagles
Philadelphia Eagles . September 9 , 2016 . Retrieved September 17 , 2017 . Jump up ^ `` Philadelphia Eagles Team Capsule '' ( PDF ) . 2017 Official National
Football League Record and Fact Book . National Football League . August 22 , 2017 . Retrieved September 17 , 2017 . Jump up ^ `` Top 10 NFL Rivalries Of All
Time : No. 4 Giants - Eagles '' . Sports Illustrated . December 15 , 2005 . Retrieved September 17 , 2017 . Jump up ^ Chadiha , Jeffri ( October 31 , 2007 ) .
`` Ranking the NFL 's best rivalries : Where does Colts - Pats fit ? '' . ESPN.com . Retrieved April 12 , 2008 . Jump up ^ Bryan , Dave ( September 20 , 2016 )
. `` After 8 Straight Losses , Steelers Looking For Philadelphia Freedom Sunday Against Eagles '' . SteelersDepot.com . Retrieved September 17 , 2017 .
</item 5>

<item 6>
New England fell behind for the first time in its postseason run. After Corey Dillon helped drive the Patriots to Philadelphia’s 4-yard line, Tom Brady fumbled
and Eagles defender Darwin Walker recovered. Philadelphia went three-and-out, and a 29-yard punt gave New England possession at the Eagles’ 37. Brady then threw
a 4-yard touchdown to David Givens with 1:10 left in the half, tying the score 7–7. It was only the second halftime tie
</item 6>

<item 7>
Philadelphia Eagles
Season records Main article : List of Philadelphia Eagles seasons Regular season record ( all - time ) : 568 -- 594 -- 26 Playoff record ( all - time ) : 22 --
21 ( last appearance after beating New England Patriots in Super Bowl LII ) Most points in a season : 474 points ( 2014 ) NFL championships won : 4 ( 3 before
the 1967 NFL - AFL merger created the Super Bowl ) Super Bowls won : 1 out of 3 appearances Passing leader ( all - time ) : Donovan McNabb -- 32,873 yards
Rushing leader ( all - time ) : LeSean McCoy -- 6,792 yards Receiving leader ( all - time ) : Harold Carmichael -- 8,978 yards Winningest coach ( all - time ) :
Andy Reid -- 130 wins Top player by approximate value ( all - time ) : Donovan McNabb -- 126 AV Rivalries New York Giants Main article : Eagles -- Giants
rivalry One of the NFL 's oldest , this rivalry began on October 15 , 1933 when the Giants defeated the newly founded Eagles 56 -- 0 .
</item 7>

<item 8>
On February 6, 2005, the New England Patriots defeated the Philadelphia Eagles 24–21 at Jacksonville’s Alltel Stadium, earning their third Super Bowl title in
four seasons. After a 14–14 tie entering the fourth quarter, Corey Dillon scored and Adam Vinatieri added a field goal. Donovan McNabb’s late touchdown to Greg
Lewis brought Philadelphia within three, but the Patriots held on. McNabb threw three interceptions; Tom Brady passed for 236 yards and two touchdowns.
</item 8>

<item 9>
The Philadelphia Eagles, founded in 1933 and based in Philadelphia, Pennsylvania, have played at several stadiums, including Lincoln Financial Field. They have
won 13 division titles, four conference championships (1960, 1980, 2017 and others listed), and four league championships: 1948, 1949, 1960 and Super Bowl LII
in 2017. Their rivalries include the Cowboys, Giants, Washington and Steelers. The franchise is known for traditions, memorable
</item 9>

<item 10>
Super Bowl XXXIX
For the Eagles , with the loss , the city of Philadelphia would not see its first championship since the 76ers swept the 1983 NBA Finals until the next time
their city 's teams played for a championship , when the Phillies won the 2008 World Series . The Eagles did not return to the Super Bowl until 2017 , also with
a 13 - 3 record , and defeating the Falcons and Vikings in the playoffs . The Eagles and the Patriots met again in Super Bowl LII , following the 2017 season ,
with the Eagles taking their revenge 41 -- 33 . This is the last time that a team has won back - to - back Super Bowls , with the Patriots losing their second
playoff game a year later and then , for the next eight years , every team either losing their first playoff game or missing them altogether . The streak was
broken in 2014 when the Seahawks defeated the Panthers and ultimately advanced to the Super Bowl , only to lose to a Patriots goal line stand .
</item 10>
=== END ITEM id=c27057603aad17d9 ===

=== ITEM id=4c0639e90d052d44 ===
Gold sentences and facts:
Sentence 0: We admitted we were powerless over alcohol -- that our lives had become unmanageable . Came to believe that a Power greater than ourselves could restore us to sanity . Made a decision to turn our will and our lives over to the care of God as we understood Him . Made a searching and fearless moral inventory of ourselves . Admitted to God , to ourselves , and to another human being the exact nature of our wrongs . Were entirely ready to have God remove all these defects of character . Humbly asked Him to remove our shortcomings . Made a list of all persons we had harmed , and became willing to make amends to them all . Made direct amends to such people wherever possible , except when to do so would injure them or others . Continued to take personal inventory , and when we were wrong , promptly admitted it . Sought through prayer and meditation to improve our conscious contact with God as we understood Him , praying only for knowledge of His will for us and the power to carry that out . Having had a spiritual awakening as the result of these steps , we tried to carry this message to alcoholics , and to practice these principles in all our affairs .
  Fact 0.0: We admitted we were powerless over alcohol.
  Fact 0.1: We admitted our lives had become unmanageable.
  Fact 0.2: We came to believe that a Power greater than ourselves could restore us to sanity.
  Fact 0.3: We made a decision to turn our will and our lives over to the care of God as we understood Him.
  Fact 0.4: We made a searching and fearless moral inventory of ourselves.
  Fact 0.5: We admitted to God, to ourselves, and to another human being the exact nature of our wrongs.
  Fact 0.6: We were entirely ready to have God remove all these defects of character.
  Fact 0.7: We humbly asked God to remove our shortcomings.
  Fact 0.8: We made a list of all persons we had harmed.
  Fact 0.9: We became willing to make amends to all persons we had harmed.
  Fact 0.10: We made direct amends to such people wherever possible, except when to do so would injure them or others.
  Fact 0.11: We continued to take personal inventory.
  Fact 0.12: When we were wrong, we promptly admitted it.
  Fact 0.13: We sought through prayer and meditation to improve our conscious contact with God as we understood Him.
  Fact 0.14: We prayed only for knowledge of God's will for us.
  Fact 0.15: We prayed only for the power to carry out God's will for us.
  Fact 0.16: Having had a spiritual awakening as the result of these steps, we tried to carry this message to alcoholics.
  Fact 0.17: We tried to practice these principles in all our affairs.
Sentence 1: We admitted we were powerless over alcohol -- that our lives had become unmanageable .
  Fact 1.0: We admitted we were powerless over alcohol.
  Fact 1.1: We admitted our lives had become unmanageable.

Retrieved items:
<item 1>
Twelve-step program
The A.A. Grapevine . Alcoholics Anonymous . 6 ( 6 ) . November 1949 . ISSN 0362 - 2584 . OCLC 50379271 . Jump up ^ Twelve Steps and Twelve Traditions . Hazelden
. February 2002 . ISBN 0 - 916856 - 01 - 1 . OCLC 13572433 . Jump up ^ `` For Anyone New Coming to A.A. ; For Anyone Referring People to A.A '' . Alcoholics
Anonymous World Services , Inc . Archived from the original on 2008 - 01 - 17 . Retrieved June 15 , 2006 . Jump up ^ Vaillant , George E. ( 2002 ) . ``
Singleness of Purpose '' ( PDF ) . About AA : A Newsletter for Professionals ( Fall / Winter ) . Jump up ^ Bill W. ( June 2001 ) . `` Chapter 5 : How It Works
'' . Alcoholics Anonymous ( PDF ) ( 4th ed . ) . Alcoholics Anonymous World Services .
</item 1>

<item 2>
Twelve-step program
Twelve - step program - wikipedia Twelve - step program Jump to : navigation , search A twelve - step program is a set of guiding principles outlining a course
of action for recovery from addiction , compulsion , or other behavioral problems . Originally proposed by Alcoholics Anonymous ( AA ) as a method of recovery
from alcoholism , the Twelve Steps were first published in the 1939 book Alcoholics Anonymous : The Story of How More Than One Hundred Men Have Recovered from
Alcoholism . The method was adapted and became the foundation of other twelve - step programs . As summarized by the American Psychological Association , the
process involves the following : admitting that one can not control one 's alcoholism , addiction or compulsion ; recognizing a higher power that can give
strength ; examining past errors with the help of a sponsor ( experienced member ) ; making amends for these errors ; learning to live a new life with a new
code of behavior ; helping others who suffer from the same alcoholism , addictions or compulsions .
</item 2>

<item 3>
Twelve-step program
Made a searching and fearless moral inventory of ourselves . Admitted to God , to ourselves , and to another human being the exact nature of our wrongs . Were
entirely ready to have God remove all these defects of character . Humbly asked Him to remove our shortcomings . Made a list of all persons we had harmed , and
became willing to make amends to them all . Made direct amends to such people wherever possible , except when to do so would injure them or others . Continued
to take personal inventory , and when we were wrong , promptly admitted it . Sought through prayer and meditation to improve our conscious contact with God as
we understood Him , praying only for knowledge of His will for us and the power to carry that out . Having had a spiritual awakening as the result of these
steps , we tried to carry this message to alcoholics , and to practice these principles in all our affairs . In some cases , where other twelve - step groups
have adapted the AA steps as guiding principles , step one is uniquely different for each organization , these are sometimes altered to emphasize principles
important to those particular fellowships , or to remove gender - biased language .
</item 3>

<item 4>
Twelve-step programs are recovery frameworks originating with Alcoholics Anonymous, whose steps were published in 1939 and later adapted by other groups. They
guide participants to acknowledge loss of control, seek strength from a higher power, examine past conduct with support from a sponsor, admit wrongs, make
amends, maintain personal accountability, and use prayer or meditation. The approach also emphasizes behavioral change and helping others; adapted programs may
revise steps to suit their fellowship.
</item 4>

<item 5>
Twelve-step programs originated with Alcoholics Anonymous, founded in 1935, and have since spread to hundreds of fellowships addressing addiction and other
problems. Their steps emphasize powerlessness, a Higher Power, self-examination, amends, ongoing accountability, and helping others; traditions guide
autonomous, nonprofessional groups. Members often work with sponsors. Research and criticism report mixed outcomes, and findings may not generalize across
programs. Group confidentiality is customary, not legally protected.
</item 5>

<item 6>
Twelve-step program
`` Twelve - step recovery and community service '' . Health & Social Work . 28 ( 2 ) : 137 -- 145 . doi : 10.1093 / hsw / 28.2. 137 . CS1 maint : Multiple names
: authors list ( link ) Lile , B. ( 2003 ) . `` Twelve step programs : An update '' . Addictive Disorders & Their Treatment . 2 ( 1 ) : 19 -- 24 . Morgenstern ,
J. , Bux , D. , Labouvie , E. , Blanchard , K.A. , & Morgan , T.J. ( November 2002 ) . `` Examining mechanisms of action in 12 - step treatment : The role of 12
- step cognitions '' . Journal of Studies on Alcohol . 63 ( 6 ) : 665 -- 672 . PMID 12529066 . CS1 maint : Multiple names : authors list ( link ) Sheehan , T. (
2004 ) . `` Twelve Step Facilitation : A Necessary Treatment for Offenders '' .
</item 6>

<item 7>
Twelve-step program
Contents ( hide ) 1 Overview 2 History 3 Twelve Steps 4 Twelve Traditions 5 Process 6 Sponsorship 7 Effectiveness 8 Criticism 8.1 Confidentiality 8.2 Cultural
identity 9 See also 10 References 11 Further reading 11.1 Scholarly publications Overview ( edit ) Twelve - step methods have been adapted to address a wide
range of alcoholism , substance - abuse and dependency problems . Over 200 self - help organizations -- often known as fellowships -- with a worldwide
membership of millions -- now employ twelve - step principles for recovery . Narcotics Anonymous was formed by addicts who did not relate to the specifics of
alcohol dependency . Demographic preferences related to the addicts ' drug of choice has led to the creation of Cocaine Anonymous , Crystal Meth Anonymous and
Marijuana Anonymous . Behavioral issues such as compulsion for , and / or addiction to , gambling , crime , food , sex , hoarding , debting and work are
addressed in fellowships such as Gamblers Anonymous , Overeaters Anonymous , Sexaholics Anonymous and Debtors Anonymous .
</item 7>

<item 8>
Twelve-step programs describe addiction’s mental obsession as recurring thoughts that lead people to resume compulsive behavior, often despite knowing they may
lose control. The First Step’s “unmanageability” reflects this lack of choice. The underlying spiritual malady is self-centeredness; working the steps aims to
replace it with moral awareness, self-sacrifice, and constructive, unselfish action. This gradual “spiritual awakening” differs from temporary, dramatic
abreaction. Members emphasize admitting their problem.
</item 8>

<item 9>
Twelve-step program
. Alcoholics Anonymous World Services . ISBN 1 - 893007 - 16 - 2 . OCLC 32014950 . Jump up ^ `` World Service Board of Trustees Bulletin # 13 : Some thoughts
regarding our relationship to Alcoholics Anonymous '' . Narcotics Anonymous World Services . February 23 , 2007 . Archived from the original on October 6 , 2007
. Retrieved October 7 , 2007 . Jump up ^ `` NA History Chronology '' . January 13 , 2006 . Archived from the original on March 3 , 2009 . Retrieved October 7 ,
2007 . 1953 : ... this committee met regularly ... From the beginning ... 12 Steps ... 12 Traditions ... All Steps had ' We ' Jump up ^ `` The 12 Steps of
Recovery '' . Archived from the original on 2012 - 11 - 05 . Retrieved August 28 , 2010 . Jump up ^ `` The Twelve Steps of Marijuana Anonymous '' . Retrieved
September 28 , 2012 . Jump up ^ Bill W.
</item 9>

<item 10>
Twelve-step program
As AA chapters were increasing in number during the 1930s and 1940s , the guiding principles were gradually defined as the Twelve Traditions . A singleness of
purpose emerged as Tradition Five : `` Each group has but one primary purpose -- to carry its message to the alcoholic who still suffers '' . Consequently ,
drug addicts who do not suffer from the specifics of alcoholism involved in AA hoping for recovery technically are not welcome in `` closed '' meetings unless
they have a desire to stop drinking alcohol . The principles of AA have been used to form many numbers of other fellowships specifically designed for those
recovering from various pathologies ; each emphasizes recovery from the specific malady which brought the sufferer into the fellowship . Twelve steps ( edit )
For more details on this topic , see List of Twelve Step alternate wordings . The following are the original twelve steps as published by Alcoholics Anonymous :
We admitted we were powerless over alcohol -- that our lives had become unmanageable . Came to believe that a Power greater than ourselves could restore us to
sanity . Made a decision to turn our will and our lives over to the care of God as we understood Him .
</item 10>
=== END ITEM id=4c0639e90d052d44 ===

=== ITEM id=353bc4ff582a7fc9 ===
Gold sentences and facts:
Sentence 0: In vertebrates , the gallbladder is a small hollow organ where bile is stored and concentrated before it is released into the small intestine . In humans , the pear - shaped gallbladder lies beneath the liver , although the structure and position of the gallbladder can vary significantly among animal species . It receives and stores bile , produced by the liver , via the common hepatic duct , and releases it via the common bile duct into the duodenum , where the bile helps in the digestion of fats .
  Fact 0.0: In vertebrates, the gallbladder is a small hollow organ.
  Fact 0.1: In vertebrates, bile is stored in the gallbladder before it is released into the small intestine.
  Fact 0.2: In vertebrates, bile is concentrated in the gallbladder before it is released into the small intestine.
  Fact 0.3: In humans, the gallbladder is pear-shaped.
  Fact 0.4: In humans, the gallbladder lies beneath the liver.
  Fact 0.5: The structure and position of the gallbladder can vary significantly among animal species.
  Fact 0.6: The gallbladder receives bile produced by the liver via the common hepatic duct.
  Fact 0.7: The gallbladder stores bile produced by the liver via the common hepatic duct.
  Fact 0.8: The gallbladder releases bile via the common bile duct into the duodenum.
  Fact 0.9: Bile helps in the digestion of fats in the duodenum.

Retrieved items:
<item 1>
Gallbladder
In adults , the gallbladder measures approximately 7 to 10 centimetres ( 2.8 to 3.9 inches ) in length and 4 centimetres ( 1.6 in ) in diameter when fully
distended . The gallbladder has a capacity of about 50 millilitres ( 1.8 imperial fluid ounces ) . The gallbladder is shaped like a pear , with its tip opening
into the cystic duct . The gallbladder is divided into three sections : the fundus , body , and neck . The fundus is the rounded base , angled so that it faces
the abdominal wall . The body lies in a depression in the surface of the lower liver . The neck tapers and is continuous with the cystic duct , part of the
biliary tree . The gallbladder fossa , against which the fundus and body of the gallbladder lie , is found beneath the junction of hepatic segments IVB and V .
The cystic duct unites with the common hepatic duct to become the common bile duct .
</item 1>

<item 2>
Gallbladder
5 ( 1 ) : 89 -- 94 . Books editor - in - chief , Susan Standring ; section editors , Neil R. Borley ; et al. ( 2008 ) . Gray 's anatomy : the anatomical basis
of clinical practice ( 40th ed . ) . London : Churchill Livingstone . ISBN 978 - 0 - 8089 - 2371 - 8 . CS1 maint : Explicit use of et al. ( link ) CS1 maint :
Multiple names : authors list ( link ) CS1 maint : Extra text : authors list ( link ) External links ( edit ) Wikimedia Commons has media related to Gallbladder
. Diagram of Human Stomach and Gallbladder -- Human Anatomy Online dd , MyHealthScore.com . ( hide ) Anatomy of the liver ,
</item 2>

<item 3>
Gallbladder
PMID 7482221 . Jump up ^ Segura - Sampedro , JJ ; Navarro - Sánchez , A ; Ashrafian , H ; Martínez - Isla , A ( February 2015 ) . `` Laparoscopic approach to
the intrahepatic gallbladder . A case report '' . Revista espanola de enfermedades digestivas : organo oficial de la Sociedad Espanola de Patologia Digestiva.
107 ( 2 ) : 122 -- 3 . PMID 25659400 . Archived from the original on March 4 , 2016 . Jump up ^ Dhulkotia , A ; Kumar , S ; Kabra , V ; Shukla , HS ( March 1 ,
2002 ) . `` Aberrant gallbladder situated beneath the left lobe of liver '' . HPB. 4 ( 1 ) : 39 -- 42 . doi : 10.1080 / 136518202753598726 .
</item 3>

<item 4>
Gallbladder
Society and Culture ( edit ) To have ' gall ' is associated with bold behaviour , whereas to have ' bile ' is associated with bitterness . In the Chinese
language , the gallbladder ( Chinese : 膽 ) is associated with courage and a myriad of related idioms , including using terms such as `` a body completely ( of )
gall '' ( Chinese : 渾身 是 膽 ) to describe a brave person , and `` single gallbladder hero '' ( Chinese : 孤膽 英雄 ) to describe a lone hero . In the Zangfu theory
of Chinese medicine , the gallbladder not only has a digestive role , but is seen as the seat of decision - making . Other animals ( edit ) Most vertebrates
have gallbladders , but the form and arrangement of the bile ducts may vary considerably . In many species , for example , there are several separate ducts
running to the intestine , rather than the single common bile duct found in humans . Several species of mammals ( including horses , deer , rats , and laminoids
) , several species of birds , lampreys and all invertebrates lack a gallbladder altogether .
</item 4>

<item 5>
Gallbladder
Early in development , the human embryo has three germ layers and abuts an embryonic yolk sac . During the second week of embryogenesis , as the embryo grows ,
it begins to surround and envelop portions of this sac . The enveloped portions form the basis for the adult gastrointestinal tract . Sections of this foregut
begin to differentiate into the organs of the gastrointestinal tract , such as the esophagus , stomach , and intestines . During the fourth week of
embryological development , the stomach rotates . The stomach , originally lying in the midline of the embryo , rotates so that its body is on the left . This
rotation also affects the part of the gastrointestinal tube immediately below the stomach , which will go on to become the duodenum . By the end of the fourth
week , the developing duodenum begins to spout a small outpouching on its right side , the hepatic diverticulum , which will go on to become the biliary tree .
Just below this is a second outpouching , known as the cystic diverticulum , that will eventually develop into the gallbladder . Function ( edit ) 1 .
</item 5>

<item 6>
Gallbladder
Function ( edit ) 1 . Bile ducts : 2 . Intrahepatic bile ducts , 3 . Left and right hepatic ducts , 4 . Common hepatic duct , 5 . Cystic duct , 6 . Common bile
duct , 7 . Ampulla of Vater , 8 . Major duodenal papilla 9 . Gallbladder , 10 -- 11 . Right and left lobes of liver . 12 . Spleen . 13 . Esophagus. 14 . Stomach
. 15 . Pancreas : 16 . Accessory pancreatic duct , 17 . Pancreatic duct . 18 . Small intestine : 19 . Duodenum , 20 . Jejunum 21 -- 22 . Right and left kidneys
. The front border of the liver has been lifted up ( brown arrow ) . The main purpose of the gallbladder is to store bile , also called gall , needed for the
digestion of fats in food .
</item 6>

<item 7>
Gallbladder
doi : 10.1080 / 136518202753598726 . Jump up ^ Naganuma , S. ; Ishida , H. ; Konno , K. ; Hamashima , Y. ; Hoshino , T. ; Naganuma , H. ; Komatsuda , T. ;
Ohyama , Y. ; Yamada , N. ; Ishida , J. ; Masamune , O. ( March 6 , 2014 ) . `` Sonographic findings of anomalous position of the gallbladder '' . Abdominal
Imaging. 23 ( 1 ) : 67 -- 72 . doi : 10.1007 / s002619900287 . PMID 9437066 . Jump up ^ Meilstrup JW ; Hopper KD ; Thieme GA ( December 1991 ) . `` Imaging of
gallbladder variants '' . AJR Am J Roentgenol. 157 ( 6 ) : 1205 -- 8 . doi : 10.2214 / ajr. 157.6. 1950867 . PMID 1950867 .
</item 7>

<item 8>
Gallbladder
Rarely , two or even three gallbladders may coexist , either as separate bladders draining into the cystic duct , or sharing a common branch that drains into
the cystic duct . Additionally , the gallbladder may fail to form at all . Gallbladders with two lobes separated by a septum may also exist . These
abnormalities are not likely to affect function and are generally asymptomatic . The location of the gallbladder in relation to the liver may also vary , with
documented variants including gallbladders found within , above , on the left side of , behind , and detached or suspended from the liver . Such variants are
very rare : from 1886 to 1998 , only 110 cases of left - lying liver , or less than one per year , were reported in scientific literature . An anatomical
variation can occur , known as a Phrygian cap , which is an innocuous fold in the fundus , named after its resemblance to the Phrygian cap . Development ( edit
) The gallbladder develops from an endodermal outpouching of the embryonic gut tube .
</item 8>

<item 9>
Gallbladder
Gallbladder - Wikipedia Gallbladder This is the latest accepted revision , reviewed on 18 January 2018 . Jump to : navigation , search Gallbladder The
gallbladder sits beneath the liver . Details Precursor Foregut System Digestive system Artery Cystic artery Vein Cystic vein Nerve Celiac ganglia , Vagus nerve
Identifiers Latin Vesica biliaris , vesica fellea MeSH A03. 159.439 Dorlands / Elsevier 12383343 TA A05. 8.02. 001 FMA 7202 Anatomical terminology ( edit on
Wikidata ) In vertebrates , the gallbladder is a small hollow organ where bile is stored and concentrated before it is released into the small intestine . In
humans , the pear - shaped gallbladder lies beneath the liver , although the structure and position of the gallbladder can vary significantly among animal
species . It receives and stores bile , produced by the liver , via the common hepatic duct , and releases it via the common bile duct into the duodenum , where
the bile helps in the digestion of fats .
</item 9>

<item 10>
Gallbladder
The gallbladder can be affected by gallstones , formed by material that can not be dissolved -- usually cholesterol or bilirubin , a product of haemoglobin
breakdown . These may cause significant pain , particularly in the right upper corner of the abdomen , and are often treated with removal of the gallbladder
called a cholecystectomy . Cholecystitis , inflammation of the gallbladder , has a wide range of causes , including result from the impaction of gallstones ,
infection , and autoimmune disease . Contents ( hide ) 1 Structure 1.1 Microanatomy 1.2 Variation 1.3 Development 2 Function 3 Clinical significance 3.1
Gallstones 3.2 Inflammation 3.3 Gallbladder removal 3.4 Cancer 3.5 Tests 4 Society and culture 5 Other animals 6 History 7 See also 8 References 9 External
links Structure ( edit ) The gallbladder is a hollow organ that sits in a shallow depression below the right lobe of the liver , that is grey - blue in life .
</item 10>
=== END ITEM id=353bc4ff582a7fc9 ===
