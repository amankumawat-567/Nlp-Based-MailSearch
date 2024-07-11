import re
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import calendar

class nlp_date_parser:
    def __init__(self):
        self.month_map = {
            'jan':1,'feb':2,'mar':3,'apr':4,
            'may':5,'jun':6,'jul':7,'aug':8,
            'sep':9,'oct':10,'nov':11,'dec':12,
        }
        self.weekday_map = {
            'monday':0,'tuesday':1,'wednessday':2,'thrusday':3,
            'friday':4,'saturday':5,'sunday':6,
        }
        self.date_formats = [
            "%d %m %Y", "%m %d %Y", "%Y %m %d",
            "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d",
            "%d-%m-%Y", "%m-%d-%Y", "%Y-%m-%d",
            "%B %d %Y", "%d %B %Y", "%Y-%j",
            "%Y-%W-%w", "%b %d %Y", "%d %b %Y",
            "%b %d", "%d %b", "%B %d", "%d %B",
            "%b %Y", "%Y %b","%B %Y", "%Y %B",
            "%Y"
        ]
        self.date_patterns = [
            r'(\d{1,2})\s+(days|months|years|weeks)\s*(ago|before|back)',
            r'(next|last|previous)?\s*(day|week|month|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday)',
            r'(first|second|third|fourth|last|second last)\s+(week|monday|tuesday|wednessday|thrusday|friday|saturday|sunday)\s+of\s+(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)(?:\s+(.*))?',
            r'(\d{1,2})\s+(days|weeks|months|years)\s+(before|after)(?:\s+(.*))?'
        ]
        self.get_current_date()

    def get_current_date(self):
        self.now = datetime.now()
        # Extract the components
        self.day = self.now.day
        self.month = self.now.month
        self.year = self.now.year
        self.week_number = self.now.strftime("%U")  # Week number of the year (Sunday as the first day of the week)
        self.day_of_year = self.now.timetuple().tm_yday  # Day of the year
        self.day_of_week = self.now.weekday()  # Day of the week (0 is Monday, 6 is Sunday)

        # Calculate the week of the month
        first_day_of_month = self.now.replace(day=1)
        first_day_of_month_weekday = first_day_of_month.weekday()
        self.week_of_month = (self.now.day + first_day_of_month_weekday - 1) // 7 + 1

    def direct_date(self, date_string):
        for i in range(24):
            try:
                date = datetime.strptime(date_string, self.date_formats[i])
                if i < 15:
                    return {"start_date" : date,"end_date" : date}
                day = date.day
                month = date.month
                if i < 19:
                    date = datetime(self.year,month,day)
                    return {"start_date" : date,"end_date" : date}
                year = date.year
                if i < 23:
                    return {"start_date" : datetime(year,month,1),"end_date" : datetime(year,month,calendar.monthrange(year, month)[1])}
                return {"start_date" : datetime(year,1,1),"end_date" : datetime(year,12,31)}
            except ValueError:
                continue
        return None

    def indirect_date(self, date_string):
        date_string = date_string.lower()
        if date_string == 'today':
            return {"start_date" : self.now,"end_date" : self.now}
        elif date_string == 'yesterday':
            yesterday = self.now - timedelta(days=1)
            return {"start_date" : yesterday,"end_date" : yesterday}
        elif date_string == 'tomorrow':
            tomorrow = self.now + timedelta(days=1)
            return {"start_date" : tomorrow,"end_date" : tomorrow}
        elif date_string == "day before yesterday":
            day_before_yesterday = self.now - timedelta(days=2)
            return {"start_date" : day_before_yesterday,"end_date" : day_before_yesterday}
        elif date_string == "day after tomorrow":
            day_after_tomorrow = self.now + timedelta(days=2)
            return {"start_date" : day_after_tomorrow,"end_date" : day_after_tomorrow}
        else:
            for i in range(4):
                matches = re.findall(self.date_patterns[i], date_string)
                if matches:
                    if i == 0:
                        quantity, unit,_ = matches[0]
                        if quantity == "" or unit == "":
                            continue
                        quantity = int(quantity)
                        if unit in ['days','day']:
                            end_date = self.now - timedelta(days=quantity)
                            start_date = end_date
                        elif unit in ['weeks','week']:
                            end_date = self.now - timedelta(weeks=quantity)
                            start_date = end_date
                        elif unit in ['months','month']:
                            end_date = self.now - relativedelta(months=quantity)
                            start_date = end_date.replace(day=1)
                            end_date = (start_date + relativedelta(months=1)) - timedelta(days=1)
                        else:
                            end_date = self.now - relativedelta(years=quantity)
                            start_date = end_date.replace(month=1, day=1)
                            end_date = (start_date + relativedelta(years=1)) - timedelta(days=1)

                        return {"start_date": start_date, "end_date": end_date}
                    elif i == 1:
                        direction, unit = matches[0]
                        if direction == "" or unit == "":
                            continue
                        if unit == "day":
                            delta = timedelta(days=1)
                        elif unit == "week":
                            delta = timedelta(weeks=1)
                        elif unit == "month":
                            if direction == "next":
                                start_date = (self.now.replace(day=1) + timedelta(days=32)).replace(day=1)
                            elif direction in ["last", "previous"]:
                                start_date = (self.now.replace(day=1) - timedelta(days=1)).replace(day=1)
                            end_date = start_date + timedelta(days=calendar.monthrange(start_date.year, start_date.month)[1] - 1)
                            return {"start_date": start_date, "end_date": end_date}
                        elif unit == "year":
                            if direction == "next":
                                start_date = datetime(self.year + 1, 1, 1)
                            elif direction in ["last", "previous"]:
                                start_date = datetime(self.year - 1, 1, 1)
                            end_date = datetime(start_date.year, 12, 31)
                            return {"start_date": start_date, "end_date": end_date}
                        else:
                            weekday = self.weekday_map[unit]
                            days_ahead = weekday - self.day_of_week
                            if direction == "next":
                                days_ahead += 7 if days_ahead <= 0 else 0
                            elif direction in ["last", "previous"]:
                                days_ahead -= 7 if days_ahead >= 0 else 0
                            start_date = self.now + timedelta(days=days_ahead-1)
                            return {"start_date": start_date, "end_date": start_date}

                        if direction == "next":
                            start_date = self.now + delta
                        else:
                            start_date = self.now - delta

                        end_date = start_date
                        if unit == "week":
                            end_date = start_date + timedelta(days=6)

                        return {"start_date": start_date, "end_date": end_date}
                    elif i == 2:
                        quantity, unit, month, year = matches[0]
                        if year == "":
                            year = self.year
                        else:
                            year = int(year)
                        month_number = self.month_map[month[:3]]
                        first_day_of_month = datetime(year, month_number, 1)
                        last_day_of_month = datetime(year, month_number, calendar.monthrange(self.year, month_number)[1])

                        if unit.lower() == "week":
                            # Calculate week-based ranges
                            if quantity == "first":
                                start_date = first_day_of_month
                                end_date = start_date + timedelta(days=6)
                            elif quantity == "second":
                                start_date = first_day_of_month + datetime.timedelta(days=7)
                                end_date = start_date + timedelta(days=6)
                            elif quantity == "third":
                                start_date = first_day_of_month + datetime.timedelta(days=14)
                                end_date = start_date + timedelta(days=6)
                            elif quantity == "fourth":
                                start_date = first_day_of_month + datetime.timedelta(days=21)
                                end_date = start_date + timedelta(days=6)
                            elif quantity == "last":
                                end_date = last_day_of_month
                                start_date = end_date - timedelta(days=6)
                            elif quantity == "second last":
                                end_date = last_day_of_month - datetime.timedelta(days=7)
                                start_date = end_date - timedelta(days=6)

                        else:
                            # Calculate day-based ranges
                            weekday_index = self.weekday_map[unit.lower()]
                            days_in_month = calendar.monthrange(year, month_number)[1]

                            if quantity == "first":
                                start_date = None
                                for day in range(1, 8):
                                    date = datetime(year, month_number, day)
                                    if date.weekday() == weekday_index:
                                        start_date = date
                                        break
                                end_date = start_date

                            elif quantity == "second":
                                count = 0
                                start_date = None
                                for day in range(1, 15):
                                    date = datetime(year, month_number, day)
                                    if date.weekday() == weekday_index:
                                        count += 1
                                        if count == 2:
                                            start_date = date
                                            break
                                end_date = start_date

                            elif quantity == "third":
                                count = 0
                                start_date = None
                                for day in range(1, 22):
                                    date = datetime(year, month_number, day)
                                    if date.weekday() == weekday_index:
                                        count += 1
                                        if count == 3:
                                            start_date = date
                                            break
                                end_date = start_date

                            elif quantity == "fourth":
                                count = 0
                                start_date = None
                                for day in range(1, 29):
                                    date = datetime(year, month_number, day)
                                    if date.weekday() == weekday_index:
                                        count += 1
                                        if count == 4:
                                            start_date = date
                                            break
                                end_date = start_date

                            elif quantity == "last":
                                start_date = None
                                for day in range(days_in_month, 0, -1):
                                    date = datetime(self.year, month_number, day)
                                    if date.weekday() == weekday_index:
                                        start_date = date
                                        break
                                end_date = start_date

                            elif quantity == "second last":
                                count = 0
                                start_date = None
                                for day in range(days_in_month, 0, -1):
                                    date = datetime(self.year, month_number, day)
                                    if date.weekday() == weekday_index:
                                        count += 1
                                        if count == 2:
                                            start_date = date
                                            break
                                end_date = start_date

                        return {"start_date": start_date, "end_date": end_date}
                    elif i == 3:
                        quantity, unit, direction, date = matches[0]
                        if quantity == "" or unit == "" or direction == "" or date == "":
                            continue
                        quantity = int(quantity)
                        date = self.direct_date(date)
                        if date == None:
                            continue
                        if direction == "after":
                            date = date['end_date']
                            if unit in ['days','day']:
                                end_date = date + timedelta(days=quantity)
                                start_date = end_date
                            elif unit in ['weeks','week']:
                                end_date = date + timedelta(weeks=quantity)
                                start_date = end_date
                            elif unit in ['months','month']:
                                end_date = date + relativedelta(months=quantity)
                                start_date = end_date.replace(day=1)
                                end_date = (start_date + relativedelta(months=1)) - timedelta(days=1)
                            else:
                                end_date = date + relativedelta(years=quantity)
                                start_date = end_date.replace(month=1, day=1)
                                end_date = (start_date + relativedelta(years=1)) - timedelta(days=1)
                        else:
                            date = date['start_date']
                            if unit in ['days','day']:
                                end_date = date - timedelta(days=quantity)
                                start_date = end_date
                            elif unit in ['weeks','week']:
                                end_date = date - timedelta(weeks=quantity)
                                start_date = end_date
                            elif unit in ['months','month']:
                                end_date = date - relativedelta(months=quantity)
                                start_date = end_date.replace(day=1)
                                end_date = (start_date + relativedelta(months=1)) - timedelta(days=1)
                            else:
                                end_date = date - relativedelta(years=quantity)
                                start_date = end_date.replace(month=1, day=1)
                                end_date = (start_date + relativedelta(years=1)) - timedelta(days=1)
                        return {"start_date": start_date, "end_date": end_date}
            return None

    def __call__(self, date_string):
        date = self.direct_date(date_string)
        if date:
            date['end_date'] += timedelta(hours=23, minutes=59, seconds=59)
            return date
        date = self.indirect_date(date_string)
        if date:
            date['end_date'] += timedelta(hours=23, minutes=59, seconds=59)
            return date
        return None